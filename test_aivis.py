"""AI-VIS test module"""
import sys
import glob

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse

from aivis import aivis, utils

AIVIS_BANDS_HIM = ['B08', 'B09', 'B10', 'B11', 'B13', 'B15', 'B16']
SAT_READER_HIM = 'ahi_hsd'

AIVIS_BANDS_GK = ['WV063', 'WV069', 'WV073', 'IR087', 'IR105', 'IR123', 'IR133']
SAT_READER_GK = 'ami_l1b'

AIVIS_BANDS_GOES = ['C08', 'C09', 'C10', 'C11', 'C13', 'C15', 'C16']
SAT_READER_GOES = 'abi_l1b'


if __name__ == '__main__':
    
    # parse command args
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='1.0', choices=['1.0', '1.5-small', '1.5-large'], help="Select the model to use, default is 1.0")
    parser.add_argument("--upscale", action="store_true", help="Use Real-ESRGAN based upscaler model")
    parser.add_argument("--half-precision", action="store_false", help="Use half precision model, not recommended")

    args = parser.parse_args()
    
    used_model = args.model
    do_upscale = args.upscale
    use_half_precision = args.half_precision
    
    import time
    time_start = time.time()
    
    files = glob.glob('./aivis/test_data/HIMAWARI/*')
    
    lat, lon = None, None
    crop_with_lonlat = False
    flip_lon = True
    sza, az = 35, 90
    auto_sza_az = False
    path_map = './aivis/basemap/landmask_hres.npz'
    
    scn2data = utils.SCENE2DATA(crop_with_lonlat=crop_with_lonlat, flip_lon=flip_lon, lon=lon, lat=lat)
    lons, lats, datas, basemap, utc_time, sza_az, sat_za_az = scn2data.get_input_from_satpy(path_map, files, SAT_READER_HIM, AIVIS_BANDS_HIM, auto_sza_az=auto_sza_az)
    
    print("Model: ", used_model)
    print("Time of data: ", utc_time.strftime("%Y/%m/%d %H:%M:%S"))
    
    if auto_sza_az:
        sza, az = sza_az
        print(sza, az)
    sat_za, sat_az = sat_za_az
    
    if not crop_with_lonlat:
        lat, lon = lats[~np.isnan(lats)].mean(), lons[~np.isnan(lons)].mean()
    
    latmin, latmax, lonmin, lonmax = (
        lat - 5, lat + 5, lon - 5, lon + 5
    )
    print("latmin: ", latmin, "latmax: ", latmax, "lonmin: ", lonmin, "lonmax: ", lonmax)
    
    time_model_start = time.time()
    ai_vis = aivis.AI_VIS(gpu_id='0')
    ai_vis.load(upscale=do_upscale, half_precision=use_half_precision, tile=0, tile_pad=10, pre_pad=10, arch=used_model)
    batch_out = ai_vis.data_to_aivis([(lons, lats, datas, basemap, sza, az, sat_za, sat_az)], batch_size=1, upscale=do_upscale)
    lons, lats, aivis = batch_out[0]
    ai_vis.release()
    time_model_end = time.time()
    
    # run time test
    #print("Model inference took:", str(time_model_end - time_model_start), "seconds")
    
    f = plt.figure(figsize=(5, 5), dpi=200)
    ax = f.add_axes([0, 0, 1, 1], projection=ccrs.Mercator(central_longitude=180))
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax.patch.set_facecolor("#FFFFFF")
    plt.axis('off')
    
    pcolor_kw = dict(cmap="gray", vmin=0, vmax=1)
    pcm = ax.pcolormesh(lons, lats, aivis, transform=ccrs.PlateCarree(), **pcolor_kw)
    
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"),
        facecolor="none",
        edgecolor='yellow',
        lw=0.5,
    )
    
    plt.savefig(
        "aivis.png",
        bbox_inches="tight",
        pad_inches=0,
        format="png"
    )
    
    plt.clf()
    plt.close('all')
    
    time_end = time.time()
    
    # run time test
    #print("Program running took:", str(time_end - time_start), "seconds")
