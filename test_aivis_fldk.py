#!/usr/bin/env python3
"""
profile_aivis.py – Run AI-VIS FLDK inference and emit a per-stage timing report.

‣ Hot sections profiled
   ├─ sun/sat geometry    (SCENE2DATA.get_msg_from_satpy)
   ├─ prep tensor         (AI_VIS._build_input_tensor)
   ├─ UNet forward        (AI_VIS._forward_batch)
   └─ feather merge       (merge_blocks)

Usage:
    python profile_aivis.py [--data ./aivis/test_data/FLDK/HIMAWARI] [--pad 20]
"""

import argparse
import datetime
import glob
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from aivis import utils_fldk as utils
from aivis import aivis as aivis_mod   # keep module handle for monkey-patching

# ───────────────────────────────────────── Timing helper ───────────────────────────────────────── #

_TIMES: dict[str, list[float]] = defaultdict(list)


def _cuda_sync_if_needed():
    """Synchronise CUDA so GPU kernels don’t leak into later measurements."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def tic(label: str):
    """Decorator: time a function and accumulate durations per label."""
    def _decorator(fn):
        def _wrapped(*args, **kwargs):
            _cuda_sync_if_needed()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            _cuda_sync_if_needed()
            _TIMES[label].append(time.perf_counter() - t0)
            return out
        _wrapped.__name__ = fn.__name__
        _wrapped.__doc__ = fn.__doc__
        return _wrapped
    return _decorator


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

# ───────────────────────────────────────── Utility functions ───────────────────────────────────── #

def get_locs(ref_lonlat, *, step=500, min_x=0, min_y=0, max_x=5500, max_y=5500):
    real_shape = ((max_x - min_x), (max_y - min_y))
    numx, numy = int(real_shape[0] / step), int(real_shape[1] / step)

    rangex = [(min_x + step * i, min_x + step * i + 500) for i in range(numx)]
    rangey = [(min_y + step * i, min_y + step * i + 500) for i in range(numy)]

    # pad edges so we always grab the trailing sliver
    if rangex[-1][-1] < max_x:
        rangex.append((max_x - 500, max_x))
    if rangey[-1][-1] < max_y:
        rangey.append((max_y - 500, max_y))
        numy += 1

    excess_x = step - rangex[-1][-1] + rangex[-2][-1]
    excess_y = step - rangey[-1][-1] + rangey[-2][-1]

    lon, lat = ref_lonlat
    real_ranges = [(cx, cy) for cx in rangex for cy in rangey]
    lon = np.array([lon[cx[0]:cx[1], cy[0]:cy[1]] for cx, cy in real_ranges])
    lat = np.array([lat[cx[0]:cx[1], cy[0]:cy[1]] for cx, cy in real_ranges])
    print(f"{_ts()} [I] real number of blocks: {len(real_ranges)}.")
    return lon, lat, real_ranges, numy, excess_x, excess_y


def crop_data(lonlat, datas, basemap, *, pad=0):
    lon_c, lat_c, ranges, numy, ex_x, ex_y = get_locs(lonlat, step=500 - pad)
    bmap_c = np.array([basemap[cx[0]:cx[1], cy[0]:cy[1]] for cx, cy in ranges])

    datas_c = [np.array([d[cx[0]:cx[1], cy[0]:cy[1]] for cx, cy in ranges]) for d in datas]
    return lon_c, lat_c, datas_c, bmap_c, numy, ex_x, ex_y


def Feather(data1, data2, *, pad=20, axis=0):
    alpha = np.linspace(0, 1, pad)
    if axis == 1:  # horizontal seam
        alpha = np.tile(alpha, (data1.shape[0], 1))
        blended = (1 - alpha) * data1[:, -pad:] + alpha * data2[:, :pad]
        return np.hstack([data1[:, :-pad], blended, data2[:, pad:]])
    else:          # vertical seam
        alpha = np.tile(alpha[:, None], (1, data1.shape[1]))
        blended = (1 - alpha) * data1[-pad:, :] + alpha * data2[:pad, :]
        return np.vstack([data1[:-pad, :], blended, data2[pad:, :]])


@tic("feather merge")
def merge_blocks(aivis_list, numy, excess_x, excess_y, pad):
    # horizontal stitching
    rows = []
    acc = None
    for i, block in enumerate(aivis_list):
        if acc is None:
            acc = block
        else:
            seam = block[:, excess_y:] if (i + 1) % numy == 0 else block
            acc = Feather(acc, seam, pad=pad, axis=1)
        if (i + 1) % numy == 0:
            rows.append(acc)
            acc = None

    # vertical stitching
    full = None
    for i, row in enumerate(rows):
        if full is None:
            full = row
        else:
            seam = row[excess_x:, :] if i == len(rows) - 1 else row
            full = Feather(full, seam, pad=pad, axis=0)
    return full

# ───────────────────────────────────────── Profiling ─────────────────────────────────────────── #

# utils.SCENE2DATA.get_msg_from_satpy = tic("sun/sat geometry")(utils.SCENE2DATA.get_msg_from_satpy)
# aivis_mod.AI_VIS._build_input_tensor = tic("prep tensor")(aivis_mod.AI_VIS._build_input_tensor)
# aivis_mod.AI_VIS._forward_batch = tic("UNet forward")(aivis_mod.AI_VIS._forward_batch)

# ───────────────────────────────────────── Main runner ────────────────────────────────────────── #

AIVIS_BANDS_HIM = ['B08', 'B09', 'B10', 'B11', 'B13', 'B15', 'B16']
SAT_READER_HIM  = 'ahi_hsd'


def run_aivis_fldk(files, *, pad=0,model="1.0", batch_size=1, half_precision=False, map_path="./aivis/basemap/himawari8.npz"):
    scn2data = utils.SCENE2DATA(crop_with_lonlat=False, flip_lon=True, lon=None, lat=None)
    lons, lats, datas, basemap, utc, sat_lon, sat_lat, sat_alt = scn2data.get_datas_from_satpy(
        map_path, files, SAT_READER_HIM, AIVIS_BANDS_HIM)

    cropped = crop_data((lons, lats), datas, basemap, pad=pad)
    lon_c, lat_c, datas_c, bmap_c, numy, ex_x, ex_y = cropped
    bt08_c, bt09_c, bt10_c, bt11_c, bt13_c, bt15_c, bt16_c = datas_c

    mask_inf = np.isinf(lons)
    print(f"{_ts()} [I] scene UTC time: {utc:%Y/%m/%d %H:%M:%S}")

    ai = aivis_mod.AI_VIS(gpu_id='0')
    ai.load(arch=model, weight_path="./aivis/weights", upscale=False, half_precision=half_precision)

    aivis_tiles = [None] * len(lon_c)
    batch_counter = 0
    
    tile_buf, pos_buf = [], []
    def _flush():
        """Forward the buffered tiles (if any) through the UNet."""
        if not tile_buf:
            return
        outs = ai.data_to_aivis(tile_buf, batch_size=len(tile_buf),
                                upscale=False)
        for (lon_, lat_, out), pos in zip(outs, pos_buf):
            aivis_tiles[pos] = out
        tile_buf.clear()
        pos_buf.clear()
        
    for idx, (lons_i, lats_i, bmap_i, bt8, bt9, bt10, bt11, bt13, bt15, bt16) in enumerate(
            zip(lon_c, lat_c, bmap_c, bt08_c, bt09_c, bt10_c, bt11_c, bt13_c, bt15_c, bt16_c)):

        datas_i = np.array([bt8, bt9, bt10, bt11, bt13, bt15, bt16])
        if np.all(np.isnan(datas_i)):
            aivis_tiles[idx] = np.zeros((500, 500))
            continue

        # solar/sat geometry
        sza, az, sat_za, sat_az = scn2data.get_msg_from_satpy(
            lons_i, lats_i, sat_lon, sat_lat, sat_alt, utc)
        
        tile_buf.append((lons_i, lats_i, datas_i, bmap_i,
                         sza, az, sat_za, sat_az))
        pos_buf.append(idx)

        batch_counter += 1
        if len(tile_buf) == batch_size:
            _flush()

    _flush() 

    ai.release()
    mosaic = merge_blocks(aivis_tiles, numy, ex_x, ex_y, pad)
    mosaic = np.where(mask_inf, 1.0, mosaic)
    return mosaic


def plot_aivis(img, out_png="aivis_fldk.png"):
    h, w = img.shape
    fig = plt.figure(figsize=(w, h), dpi=1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

"""
def print_report():
    print("\n── Profiling summary ──")
    lines = []
    for label, durs in _TIMES.items():
        lines.append((sum(durs), f"{label:16s} | calls={len(durs):4d} "
                                 f"total={sum(durs):7.3f}s  avg={np.mean(durs):6.3f}s"))
    for _, line in sorted(lines, reverse=True):
        print(line)
    print("──────────────────────\n")
    
"""


def parse_args():
    p = argparse.ArgumentParser(description="AI-VIS FLDK profiler")
    p.add_argument("--data", type=str, default="./aivis/test_data/FLDK/HIMAWARI",
                   help="glob path to Himawari HSD files")
    p.add_argument("--model", type=str, default='1.0', choices=['1.0', '1.5-small', '1.5-large'], help="Select the model to use, default is 1.0")
    p.add_argument("--upscale", action="store_false", help="Use Real-ESRGAN based upscaler model")
    p.add_argument("--pad", type=int, default=20, help="feathering pad (pixels)")
    p.add_argument("--batch-size", type=int, default=1, help="batch size for UNet forward")
    p.add_argument("--half-precision", action="store_false", help="Use half precision model, not recommended")
    p.add_argument("--output-name", type=str, default="aivis_fldk.png", help="output PNG name")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(str(Path(args.data) / "*")))
    if not files:
        sys.exit(f"No input data found under {args.data!r}")

    img = run_aivis_fldk(files, pad=args.pad,model=args.model, batch_size=args.batch_size, half_precision=args.half_precision)
    plot_aivis(img, out_png=args.output_name)
    # print_report()


if __name__ == "__main__":
    main()
