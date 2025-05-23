"""AI-VIS utils module"""
import warnings
import gc
import numpy as np
from satpy.scene import Scene
from geopy.distance import geodesic
from .astronomy import get_observer_look, get_alt_az
import dask.array as da
from scipy.interpolate import RectBivariateSpline

class InvalidAreaError(Exception):
    pass

class SCENE2DATA:
    def __init__(self, crop_with_lonlat=False, flip_lon=False, lon=None, lat=None, sat="Himawari8"):
        """Init AI-VIS utils module.

        Args:
            crop_with_lonlat (bool, optional): Whether to crop data with given longitude and latitude.
            flip_lon (bool, optional): Whether to flip longitude data. If yes, will add 360 for longitude that lower than 0, or minus 360 that greater than 180.
            lon (float, optional): Center longitude to crop data.
            lat (float, optional): Center latitude to crop data.
        """
        self.crop_with_lonlat = crop_with_lonlat
        self.flip_lon = flip_lon
        self.lon = lon
        self.lat = lat
        self.sat = sat
    
    def _proj_inverse(self, lon, lat, sat):
        """Backproject longitude/latitude into column/line.

        Args:
            lon (float): Center longitude of the cropped data.
            lat (float): Center latitude of the cropped data.
            sat_lon (float): Center longitude of the satellite.
            sat_lat (float): Center latitude of the satellite.
            sat_alt (float): Altitude of the satellite.
        """
        EARTH_CONST_1 = 0.00669438
        EARTH_CONST_2 = 0.99330562
        EARTH_POLAR_RADIUS = 6356.7523
        
        if sat == "Himawari8":
            sat_lon = 140.7
            sat_alt = 42164 - 6356.7523
        SUBLON = sat_lon
        DISTANCE = sat_alt + EARTH_POLAR_RADIUS
        LOFF = 2750.5
        LFAC = 20466275
        COFF = 2750.5
        CFAC = 20466275
        
        DEGTORAD = np.pi / 180.0
        RADTODEG = 180.0 / np.pi
        SCLUNIT = np.power(2.0, -16)
        
        lon *= DEGTORAD
        lat *= DEGTORAD
        phi = np.arctan(EARTH_CONST_2 * np.tan(lat))
        Re = EARTH_POLAR_RADIUS / np.sqrt(1 - EARTH_CONST_1 * np.square(np.cos(phi)))
        r1 = DISTANCE - Re * np.cos(phi) * np.cos(lon - SUBLON * DEGTORAD)
        r2 = -Re * np.cos(phi) * np.sin(lon - SUBLON * DEGTORAD)
        r3 = Re * np.sin(phi)
        # seeable = r1 * (r1 - _data_['Distance']) + np.square(r2) + np.square(r3)
        rn = np.sqrt(np.square(r1) + np.square(r2) + np.square(r3))
        x = np.arctan2(-r2, r1) * RADTODEG
        y = np.arcsin(-r3 / rn) * RADTODEG
        column = COFF + x * SCLUNIT * CFAC
        line = LOFF + y * SCLUNIT * LFAC
        return column, line
    
    def _proj_inverse_basemap(self, map_shape, lons, lats):
        """Backproject longitude/latitude into column/row to crop basemap.

        Args:
            map_shape (Tuple(int)): shape of map.
            lons (np.ndarray): Input longitude data to crop.
            lats (np.ndarray): Input latitude data to crop.
        """
        row, col = map_shape
        max_lat, max_lon = 180, 360
        
        _lons, _lats = lons.copy(), lats.copy()
        _lons[_lons<0] += 360
        _lats[:] += 90
        
        rows = row * (1 - _lats / max_lat)
        cols = col * (_lons / max_lon)
        
        rows[(np.isnan(rows)) | (np.isinf(rows))] = 0
        cols[(np.isnan(cols)) | (np.isinf(cols))] = 0
        
        return cols.astype(int), rows.astype(int)
    
    def get_area_bound(self, latmin, latmax, lonmin, lonmax):
        # Left edge: (latmin, lonmin) -> (latmax, lonmin)
        buffer = 3
        left_line_lat = np.linspace(latmin, latmax, 50)
        left_line_lon = np.ones_like(left_line_lat) * lonmin
        lin, pix = SCENE2DATA._proj_inverse(self, left_line_lon, left_line_lat, self.sat)
        y_min = lin.min() - buffer
        # Right edge: (latmin, lonmax) -> (latmax, lonmax)
        right_line_lat = np.linspace(latmin, latmax, 50)
        right_line_lon = np.ones_like(left_line_lat) * lonmax
        lin, pix = SCENE2DATA._proj_inverse(self, right_line_lon, right_line_lat, self.sat)
        y_max = lin.max() + buffer
        # Top edge: (latmax, lonmin) -> (latmax, lonmax)
        top_line_lon = np.linspace(lonmin, lonmax, 50)
        top_line_lat = np.ones_like(top_line_lon) * latmax
        lin, pix = SCENE2DATA._proj_inverse(self, top_line_lon, top_line_lat, self.sat)
        # Line number increases from top to bottom
        x_min = pix.min() - buffer
        # Bottom edge: (latmin, lonmin) -> (latmin, lonmax)
        bottom_line_lon = np.linspace(lonmin, lonmax, 50)
        bottom_line_lat = np.ones_like(bottom_line_lon) * latmin
        lin, pix = SCENE2DATA._proj_inverse(self, bottom_line_lon, bottom_line_lat, self.sat)
        x_max = pix.max() + buffer
        return (int(round(x_min)), int(round(x_max)), int(round(y_min)), int(round(y_max)))
    
    def _get_basemap(self, path_map, lons, lats):
        """Get basemap based on cropped area.
        If map data is not found, will return an array filled with 0.

        Args:
            path_map (str): Path of the map data to read (need a `.npz` file).
            lons (np.ndarray): Input longitude data to crop.
            lats (np.ndarray): Input latitude data to crop.
        """
        if not path_map.endswith('.npz'):
            raise ValueError('must give a .npz file to read.')
        
        try:
            with np.load(path_map) as npLoad:
                basemap = npLoad['basemap']
                map_shape = basemap.shape
            cols, rows = self._proj_inverse_basemap(map_shape, lons, lats)
            basemap = basemap[rows, cols]
        except BaseException:
            basemap = np.zeros((500, 500))
        
        return basemap

    def _get_fldkbasemap(self, path_map):
        if not path_map.endswith('.npz'):
            raise ValueError('must give a .npz file to read.')
        with np.load(path_map) as npLoad:
            basemap = npLoad['arr']
        return basemap

    def _get_band_data(self, ds, channels):
        """Get single channel data using Satpy modules.

        Args:
            ds (xarray.Dataset): xr.Dataset object transformed from Satpy.scene object.
            channels (List[str]): List of channels to get data.
        """
        datas = []
        for channel in channels:
            data = ds[channel].values
            datas.append(data - 273.15)
        datas = np.asarray(datas)
        return datas
    
    def get_datas_from_satpy(self, path_map, files, reader, load_channels):
        # load channels
        scn = Scene(files, reader=reader)
        scn.load(load_channels)
        ds = scn.to_xarray_dataset()
        del scn; gc.collect()
        
        # get satellite meta
        utc_time = ds.start_time
        sat_lon = ds[load_channels[0]].orbital_parameters['projection_longitude']
        sat_lat = ds[load_channels[0]].orbital_parameters['projection_latitude']
        sat_alt = ds[load_channels[0]].orbital_parameters['projection_altitude']
        sat_alt = sat_alt / 1000
        
        # extract all data
        datas = self._get_band_data(ds, load_channels)
        lons, lats = ds[load_channels[0]].attrs['area'].get_lonlats()

        if reader=='ahi_hsd':
            basemap = self._get_fldkbasemap(path_map)
        else:
            basemap = self._get_basemap(path_map, lons, lats)
        del ds; gc.collect()
        
        return lons, lats, datas, basemap, utc_time, sat_lon, sat_lat, sat_alt
    
    def get_msg_from_satpy(self, lon, lat, sat_lon, sat_lat, sat_alt, utc_time):
        # calculate `sza` and `az`
        salt, saz = get_alt_az(utc_time, lon, lat)
        sza = 90 - np.absolute(salt * 180 / np.pi)
        az = saz * 180 / np.pi
        
        # calculate `sat_az` and `sat_za`
        sat_az, sat_el = get_observer_look(sat_lon, sat_lat, sat_alt, utc_time, lon, lat, 0)
        sat_za = 90 - sat_el
        
        return sza, az, sat_za, sat_az

def Interp(*fields, dx=1/4, dy=1/4):
    ny, nx = fields[0].shape
    x = np.arange(nx)
    y = np.arange(ny)
    newx = np.arange(0, nx, dx)
    newy = np.arange(0, ny,dy)
    out = []
    for idx, arr in enumerate(fields):
        spline = RectBivariateSpline(y, x, arr)
        interp = spline(newy, newx)
        out.append(interp)
    return out[0] if len(out) == 1 else tuple(out)