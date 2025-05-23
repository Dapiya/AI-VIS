"""AI-VIS utils module"""
import warnings
import gc
import numpy as np
from satpy.scene import Scene
from geopy.distance import geodesic
from .astronomy import get_observer_look, get_alt_az, sun_zenith_angle

class InvalidAreaError(Exception):
    pass

class SCENE2DATA:
    def __init__(self, crop_with_lonlat=False, flip_lon=False, lon=None, lat=None):
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
    
    @staticmethod
    def _get_indices(georange, lat, lon):
        latmin, latmax, lonmin, lonmax = georange
        barr = (
                (lat > latmin - 0.5)
                & (lat < latmax + 0.5)
                & (lon > lonmin - 0.5)
                & (lon < lonmax + 0.5)
        )
        barrind = da.where(barr)
        barrind_y = barrind[0].compute()
        barrind_x = barrind[1].compute()
        yi, yj = np.amin(barrind_y), np.amax(barrind_y)
        xi, xj = np.amin(barrind_x), np.amax(barrind_x)
        return (yi, yj, xi, xj)

    def _proj_inverse(self, lon, lat, sat_lon, sat_lat, sat_alt):
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
    
    def _crop_data(self, sat_lon, sat_lat, sat_alt, lons, lats, datas):
        """Crop longitude data, latitude data, and IR data and use to AI-VIS model.
        Will give a warning if the distance between given location and after-cropped center location is longer than 120km.
        Raised `ValueError` when the distance is longer than 350km.

        Args:
            sat_lon (float): Center longitude of the satellite.
            sat_lat (float): Center latitude of the satellite.
            sat_alt (float): Altitude of the satellite.
            lons (np.ndarray): Input longitude data to crop.
            lats (np.ndarray): Input latitude data to crop.
            datas (np.ndarray): Input infrared data to crop.
        """
        if not len(datas.shape) == 3:
            raise ValueError("data should be a list of 2D-arrays")
        
        if lons.shape[0] == lats.shape[0] == datas.shape[1] < 500 or \
           lons.shape[1] == lats.shape[1] == datas.shape[2] < 500:
            raise ValueError("Size of any dimension of 2D-array must equal or larger than 500")
        
        lons[np.isinf(lons)] = np.nan
        lats[np.isinf(lats)] = np.nan
        
        if self.flip_lon:
            if self.lon is not None and self.lon < 0:
                self.lon += 360
            lons[lons<0] += 360
        else:
            if self.lon is not None and self.lon > 180:
                self.lon -= 360
            lons[lons>180] -= 360
        
        if not self.crop_with_lonlat:
            centery, centerx = int(lats.shape[0] / 2), int(lons.shape[1] / 2)
        else:
            if self.lon is None or self.lat is None:
                raise ValueError("lon/lat value is not None")
            
            column, line = self._proj_inverse(self.lon, self.lat, sat_lon, sat_lat, sat_alt)
            x, y = int(column), int(line)
            
            if y < 250:
                y = 250
            elif y > lats.shape[0] - 250:
                y = lats.shape[0] - 250
            
            if x < 250:
                x = 250
            elif x > lons.shape[1] - 250:
                x = lons.shape[1] - 250
            
            centery, centerx = y, x
            
            desic = np.absolute(geodesic((self.lat, self.lon), (lats[centery][centerx], lons[centery][centerx])).km)
            
            if desic > 350:
                raise ValueError(
                    f'\nInput lonlat: {self.lat} {self.lon}\n'
                    f'Output center lonlat: {lats[centery][centerx]} {lons[centery][centerx]}\n'
                    f'Distance offset: {desic} km\n'
                    'Offset distance is too long. Please check your input.'
                )
            elif 120 <= desic <= 350:
                warnings.warn(
                    f'\nInput lonlat: {self.lat} {self.lon}\n'
                    f'Output center lonlat: {lats[centery][centerx]} {lons[centery][centerx]}\n'
                    f'Distance offset: {desic} km\n'
                    'Offset distance is longer than 120km. It may cause some problems.'
                )
        
        lons = lons[centery-250:centery+250,centerx-250:centerx+250]
        lats = lats[centery-250:centery+250,centerx-250:centerx+250]
        datas = np.array([data[centery-250:centery+250,centerx-250:centerx+250] for data in datas])
        
        if True in np.isnan(datas[0]):
            raise InvalidAreaError('Too fringe for cropped area.')
        
        return lons, lats, datas
    
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
    
    def get_input_from_satpy(self, path_map, files, reader, load_channels, auto_sza_az=False):
        """Get input data for AI-VIS model using Satpy modules.

        Args:
            path_map (str): Path of the map to read.
            files (List[str]): Files of specific satellite data.
            reader (str): Reader for reading specific satellite data.
            load_channels (str): Channels to get data.
            auto_sza_az (bool, optional): Whether to calculate sun zenith & azimuth angle automatically.
        """
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
        lons, lats, datas = self._crop_data(sat_lon, sat_lat, sat_alt, lons, lats, datas)
        basemap = self._get_basemap(path_map, lons, lats)
        
        del ds; gc.collect()
        
        sza, az = None, None
        # calculate `sza` and `az` automatically when these are None
        if auto_sza_az:
            _, saz = get_alt_az(utc_time, lons, lats)
            sza = sun_zenith_angle(utc_time, lons, lats)
            az = np.rad2deg(saz)
        
        # calculate `sat_az` and `sat_za`
        sat_az, sat_el = get_observer_look(sat_lon, sat_lat, sat_alt, utc_time, lons, lats, 0)
        sat_za = 90 - sat_el
        
        return lons, lats, datas, basemap, utc_time, (sza, az), (sat_za, sat_az)
    
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
        basemap = self._get_basemap(path_map, lons, lats)
        
        del ds; gc.collect()
        
        return lons, lats, datas, basemap, utc_time, sat_lon, sat_lat, sat_alt
    
    def get_msg_from_satpy(self, lon, lat, sat_lon, sat_lat, sat_alt, utc_time):
        # calculate `sza` and `az`
        _, saz = get_alt_az(utc_time, lon, lat)
        sza = sun_zenith_angle(utc_time, lon, lat)
        az = np.rad2deg(saz)
        
        # calculate `sat_az` and `sat_za`
        sat_az, sat_el = get_observer_look(sat_lon, sat_lat, sat_alt, utc_time, lon, lat, 0)
        sat_za = 90 - sat_el
        
        return sza, az, sat_za, sat_az
