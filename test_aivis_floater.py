"""AI-VIS test module"""
import sys
import glob
import datetime
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from aivis import utils_fldk as utils
from aivis import aivis

AIVIS_BANDS_HIM = ['B08', 'B09', 'B10', 'B11', 'B13', 'B15', 'B16']
SAT_READER_HIM = 'ahi_hsd'

AIVIS_BANDS_GK = ['WV063', 'WV069', 'WV073', 'IR087', 'IR105', 'IR123', 'IR133']
SAT_READER_GK = 'ami_l1b'

AIVIS_BANDS_GOES = ['C08', 'C09', 'C10', 'C11', 'C13', 'C15', 'C16']
SAT_READER_GOES = 'abi_l1b'

def get_time_string():
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

def get_locs(ref_lonlat, step=500, min_x=0, min_y=0, max_x=5500, max_y=5500):

    real_shape = ( (max_x - min_x) , (max_y - min_y) )
    numx, numy = int(real_shape[0] / step), int(real_shape[1] / step)
    
    rangex = [(min_x + step * i, min_x + step * i + 500) for i in range(numx)]
    rangey = [(min_y + step * i, min_y + step * i + 500) for i in range(numy)]
    if rangex[-1][-1] < max_x:
        rangex.append((max_x-500, max_x))
    if rangey[-1][-1] < max_y:
        rangey.append((max_y-500, max_y))
        numy += 1
    excess_x = step - rangex[-1][-1] + rangex[-2][-1]
    excess_y = step - rangey[-1][-1] + rangey[-2][-1]

    lon, lat = ref_lonlat
    real_ranges = []
    for cx in rangex:
        for cy in rangey:
            real_ranges.append((cx, cy))
    lon = np.array([lon[cx[0]:cx[1],cy[0]:cy[1]] for cx, cy in real_ranges])
    lat = np.array([lat[cx[0]:cx[1],cy[0]:cy[1]] for cx, cy in real_ranges])
    print(f'{get_time_string()} [I] real number of blocks: {len(real_ranges)}.')
    return lon, lat, real_ranges, numy, excess_x, excess_y

def crop_data(lonlat, datas, basemap, rmin, rmax, cmin, cmax, pad=0):
    mp = 1

    lon_cropped, lat_cropped, ranges, numy, excess_x, excess_y = get_locs(lonlat, step=500-pad, min_x=rmin, max_x=rmax, min_y=cmin, max_y=cmax)
    basemap_cropped = np.array([basemap[int(cx[0]*mp):int(cx[1]*mp),int(cy[0]*mp):int(cy[1]*mp)] for cx, cy in ranges])
    
    datas_cropped = []
    for data in datas:
        dc = np.array([data[int(cx[0]*mp):int(cx[1]*mp),int(cy[0]*mp):int(cy[1]*mp)] for cx, cy in ranges])
        datas_cropped.append(dc)
    
    return lon_cropped, lat_cropped, datas_cropped, basemap_cropped, numy, excess_x, excess_y

def Feather(data1, data2, pad = 20, axis=0):
    alpha = np.linspace(0, 1, pad)
    if axis == 1:
        alpha = np.tile(alpha, (np.shape(data1)[0], 1))
        overlap1 = data1[:, -pad:]
        overlap2 = data2[:, :pad]
        limit = np.shape(overlap2)[1]
        blended_overlap = (1 - alpha[:, :limit]) * overlap1[:, :limit] + alpha[:, :limit] * overlap2
        blended_image = np.hstack([data1[:, :-pad], blended_overlap, data2[:, pad:]])
    else:
        alpha = np.tile(alpha[:, np.newaxis], (1, np.shape(data1)[1]))
        overlap1 = data1[-pad:, :]
        overlap2 = data2[:pad, :]
        limit = np.shape(overlap2)[0]
        blended_overlap = (1 - alpha[:limit, :]) * overlap1[:limit, :] + alpha[:limit, :] * overlap2
        blended_image = np.vstack([data1[:-pad, :], blended_overlap, data2[pad:, :]])
    return blended_image

def merge_blocks(aivis_list, numy, excess_x, excess_y, pad, upscale_f):
    aivis = np.array([])
    outs = []
    for i in range(len(aivis_list)):
        if aivis.size == 0:
            aivis = aivis_list[i]
        else:
            aivis = Feather(aivis, aivis_list[i][:, excess_y*upscale_f:] if (i + 1) % numy == 0 else aivis_list[i], pad*upscale_f, axis=1)
        if (i + 1) % numy == 0:
            outs.append(aivis)
            aivis = np.array([])
    aivis = np.array([]) # set it to avoid bug
    for i in range(len(outs)):
        if aivis.size == 0:
            aivis = outs[i]
        else:
            aivis = Feather(aivis, outs[i][excess_x*upscale_f:, :] if i == len(outs)-1 else outs[i], pad*upscale_f, axis=0)
    return aivis

def run_aivis_floater(files, lonc, latc, lonr, latr, upscale=True, center=True, path_map='./aivis/basemap/himawari8.npz', pad=0, model='1.0'):
    lat, lon = None, None
    crop_with_lonlat = False
    flip_lon = True
    sat="Himawari8"

    scn2data = utils.SCENE2DATA(crop_with_lonlat=crop_with_lonlat, flip_lon=flip_lon, lon=lon, lat=lat, sat=sat)
    lons, lats, datas, basemap, utc_time, sat_lon, sat_lat, sat_alt = scn2data.get_datas_from_satpy(path_map, files, SAT_READER_HIM, AIVIS_BANDS_HIM)

    rmin, rmax, cmin, cmax = scn2data.get_area_bound(latc-latr, latc+latr, lonc-lonr, lonc+lonr)
    cc, rc = scn2data._proj_inverse(lonc, latc, scn2data.sat)
    r_off, c_off = 0, 0
    if center:
        r_off = (250 - (int(round(rc)) - rmin) % (500 - pad)) % (500 - pad)
        c_off = (250 - (int(round(cc)) - cmin) % (500 - pad)) % (500 - pad)
        rmin -= r_off
        cmin -= c_off
    lon_cropped, lat_cropped, datas_cropped, basemap_cropped, numy, excess_x, excess_y = crop_data((lons, lats), datas, basemap, rmin, rmax, cmin, cmax, pad)
    bt08_cropped, bt09_cropped, bt10_cropped, bt11_cropped, bt13_cropped, bt15_cropped, bt16_cropped = datas_cropped
    
    print(utc_time.strftime("%Y/%m/%d %H:%M:%S"))

    ai_vis = aivis.AI_VIS(gpu_id='0')
    ai_vis.load(arch=model, weight_path='./aivis/weights', upscale=upscale)
    aivis_list = []
    for _lons, _lats, bmap, bt08, bt09, bt10, bt11, bt13, bt15, bt16 in zip(
        lon_cropped, lat_cropped, basemap_cropped,
        bt08_cropped, bt09_cropped, bt10_cropped, bt11_cropped, bt13_cropped, bt15_cropped, bt16_cropped
    ):
        datas = np.array([bt08, bt09, bt10, bt11, bt13, bt15, bt16])
        if np.all(np.isnan(datas)):
            out = np.zeros((500, 500))
        else:
            sza, az, sat_za, sat_az = scn2data.get_msg_from_satpy(_lons, _lats, sat_lon, sat_lat, sat_alt, utc_time)
            batch_out = ai_vis.data_to_aivis([(_lons, _lats, datas, bmap, sza, az, sat_za, sat_az)], batch_size=1, upscale=upscale)
            lons_out, lats_out, out = batch_out[0]
        aivis_list.append(out)
    ai_vis.release()
    upscale_f = 4 if upscale else 1
    data = merge_blocks(aivis_list, numy, excess_x, excess_y, pad, upscale_f)

    start_r = r_off * upscale_f
    start_c = c_off * upscale_f
    data = data[start_r:, start_c:]

    rmin += r_off
    cmin += c_off
    lons = lons[rmin:rmax, cmin:cmax]
    lats = lats[rmin:rmax, cmin:cmax]

    from skimage.transform import resize
    if upscale:
        data_interp = data
        if data_interp.shape != lons.shape:
            data = resize(data_interp, lons.shape, order=1, preserve_range=True, anti_aliasing=True)
        else:
            data = data_interp
    return data, lons, lats, lonc, latc, lonr, latr

def plot_aivis(data, lon, lat, lonc, latc, lonr, latr):
    f = plt.figure(figsize=(lonr*2, latr*2), dpi=150)
    ax = f.add_axes([0, 0, 1, 1])
    ax.patch.set_facecolor("#FFFFFF")
    plt.axis('off')
    ax.pcolormesh(lon, lat, data, vmin=0, vmax=1, cmap='gray')
    plt.xlim(lonc-lonr, lonc+lonr)
    plt.ylim(latc-latr, latc+latr)
    plt.savefig(
        "aivis_fldk_floater2.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png"
    )
    plt.clf()
    plt.close('all')
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="AI-VIS test module")
    parser.add_argument("--model", type=str, default='1.0', choices=['1.0', '1.5-small', '1.5-large'], help="Select the model to use, default is 1.0")
    parser.add_argument("--upscale", action="store_false", help="Use Real-ESRGAN based upscaler model")
    parser.add_argument("--half-precision", action="store_false", help="Use half precision model, not recommended")
    parser.add_argument("--pad", type=int, default=20, help="feathering pad (pixels)")
    parser.add_argument("--data", type=str, default="./aivis/test_data/FLDK/HIMAWARI", help="glob path to Himawari HSD files")
    parser.add_argument('--lonc', type=float, default=140, help='Center longitude')
    parser.add_argument('--latc', type=float, default=15, help='Center latitude')
    parser.add_argument('--lonr', type=float, default=10, help='Longitude range')
    parser.add_argument('--latr', type=float, default=10, help='Latitude range')
    parser.add_argument('--center', action='store_true', help='Center the output')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    files = glob.glob(args.data + '/*')
    data, lon, lat, lonc, latc, lonr, latr = run_aivis_floater(files, lonc=args.lonc, latc=args.latc, lonr=args.lonr, latr=args.latr,
                          upscale=args.upscale, center=args.center, pad=args.pad, model=args.model)
    plot_aivis(data, lon, lat, lonc, latc, lonr, latr)