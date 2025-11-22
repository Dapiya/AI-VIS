import numpy as np
import dask.array as da


F = 1 / 298.257223563  # Earth flattening WGS-84
A = 6378.137  # WGS84 Equatorial radius
MFACTOR = 7.292115e-5


def dt2np(utc_time):
    try:
        return np.datetime64(utc_time)
    except ValueError:
        return utc_time.astype("datetime64[ns]")


def _days(dt):
    """Get the days (floating point) from *d_t*.
    """
    return dt / np.timedelta64(1, "D")


def jdays2000(utc_time):
    """Get the days since year 2000.
    """
    return _days(dt2np(utc_time) - np.datetime64("2000-01-01T12:00"))


def sun_ecliptic_longitude(utc_time):
    """Ecliptic longitude of the sun at *utc_time*.
    """
    jdate = jdays2000(utc_time) / 36525.0
    # mean anomaly, rad
    m_a = np.deg2rad(
        357.52910
        + 35999.05030 * jdate
        - 0.0001559 * jdate * jdate
        - 0.00000048 * jdate * jdate * jdate
    )
    # mean longitude, deg
    l_0 = 280.46645 + 36000.76983 * jdate + 0.0003032 * jdate * jdate
    d_l = (
        (1.914600 - 0.004817 * jdate - 0.000014 * jdate * jdate) * np.sin(m_a)
        + (0.019993 - 0.000101 * jdate) * np.sin(2 * m_a)
        + 0.000290 * np.sin(3 * m_a)
    )
    # true longitude, deg
    l__ = l_0 + d_l
    return np.deg2rad(l__)


def sun_ra_dec(utc_time):
    """Right ascension and declination of the sun at *utc_time*.
    """
    jdate = jdays2000(utc_time) / 36525.0
    eps = np.deg2rad(
        23.0
        + 26.0 / 60.0
        + 21.448 / 3600.0
        - (46.8150 * jdate + 0.00059 * jdate * jdate - 0.001813 * jdate * jdate * jdate)
        / 3600
    )
    eclon = sun_ecliptic_longitude(utc_time)
    x__ = np.cos(eclon)
    y__ = np.cos(eps) * np.sin(eclon)
    z__ = np.sin(eps) * np.sin(eclon)
    r__ = np.sqrt(1.0 - z__ * z__)
    # sun declination
    declination = np.arctan2(z__, r__)
    # right ascension
    right_ascension = 2 * np.arctan2(y__, (x__ + r__))
    return right_ascension, declination


def _local_hour_angle(utc_time, longitude, right_ascension):
    """Hour angle at *utc_time* for the given *longitude* and
    *right_ascension*
    longitude in radians
    """
    return _lmst(utc_time, longitude) - right_ascension


def _lmst(utc_time, longitude):
    """Local mean sidereal time, computed from *utc_time* and *longitude*.
    In radians.
    """
    return gmst(utc_time) + longitude


def gmst(utc_time):
    """Greenwich mean sidereal utc_time, in radians.

    As defined in the AIAA 2006 implementation:
    http://www.celestrak.com/publications/AIAA/2006-6753/
    """
    ut1 = jdays2000(utc_time) / 36525.0
    theta = 67310.54841 + ut1 * (
        876600 * 3600 + 8640184.812866 + ut1 * (0.093104 - ut1 * 6.2 * 10e-6)
    )
    return np.deg2rad(theta / 240.0) % (2 * np.pi)


def cos_zen(utc_time, lon, lat):
    """Cosine of the sun-zenith angle for *lon*, *lat* at *utc_time*.
    utc_time: datetime.datetime instance of the UTC time
    lon and lat in degrees.
    """
    lon = da.deg2rad(lon)
    lat = da.deg2rad(lat)

    r_a, dec = sun_ra_dec(utc_time)
    h__ = _local_hour_angle(utc_time, lon, r_a)
    return da.sin(lat) * da.sin(dec) + da.cos(lat) * da.cos(dec) * da.cos(h__)


def get_alt_az(utc_time, lon, lat):
    """Return sun altitude and azimuth from *utc_time*, *lon*, and *lat*.
    lon,lat in degrees
    What is the unit of the returned angles and heights!? FIXME!
    """
    lon = da.deg2rad(lon)
    lat = da.deg2rad(lat)

    ra_, dec = sun_ra_dec(utc_time)
    h__ = _local_hour_angle(utc_time, lon, ra_)
    return (
        da.arcsin(da.sin(lat) * np.sin(dec) + da.cos(lat) * np.cos(dec) * np.cos(h__)),
        da.arctan2(
            -np.sin(h__), (da.cos(lat) * np.tan(dec) - da.sin(lat) * np.cos(h__))
        ),
    )


def sun_zenith_angle(utc_time, lon, lat):
    """Sun-zenith angle for *lon*, *lat* at *utc_time*.
    lon,lat in degrees.
    The angle returned is given in degrees
    """
    return da.rad2deg(da.arccos(cos_zen(utc_time, lon, lat)))


def get_observer_look(sat_lon, sat_lat, sat_alt, utc_time, lon, lat, alt):
    """Calculate observers look angle to a satellite.
    http://celestrak.com/columns/v02n02/

    utc_time: Observation time (datetime object)
    lon: Longitude of observer position on ground in degrees east
    lat: Latitude of observer position on ground in degrees north
    alt: Altitude above sea-level (geoid) of observer position on ground in km

    Return: (Azimuth, Elevation)
    """
    (pos_x, pos_y, pos_z), (vel_x, vel_y, vel_z) = observer_position(
        utc_time, sat_lon, sat_lat, sat_alt
    )

    (opos_x, opos_y, opos_z), (ovel_x, ovel_y, ovel_z) = observer_position(
        utc_time, lon, lat, alt
    )

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    # Compute gmst once and reuse
    gmst_val = gmst(utc_time)
    theta = (gmst_val + lon_rad) % (2 * np.pi)

    rx = pos_x - opos_x
    ry = pos_y - opos_y
    rz = pos_z - opos_z

    # Cache trig values to avoid recomputation
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    top_s = sin_lat * cos_theta * rx + sin_lat * sin_theta * ry - cos_lat * rz
    top_e = -sin_theta * rx + cos_theta * ry
    top_z = cos_lat * cos_theta * rx + cos_lat * sin_theta * ry + sin_lat * rz

    az_ = np.arctan(-top_e / top_s)

    if hasattr(az_, "chunks"):
        # dask array
        import dask.array as da

        az_ = da.where(top_s > 0, az_ + np.pi, az_)
        az_ = da.where(az_ < 0, az_ + 2 * np.pi, az_)
    else:
        az_[top_s > 0] += np.pi
        az_[az_ < 0] += 2 * np.pi

    rg_ = np.sqrt(rx * rx + ry * ry + rz * rz)
    el_ = np.arcsin(top_z / rg_)

    return np.rad2deg(az_), np.rad2deg(el_)


def observer_position(time, lon, lat, alt):
    """Calculate observer ECI position.

    http://celestrak.com/columns/v02n03/
    """

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    gmst_val = gmst(time)
    theta = (gmst_val + lon_rad) % (2 * np.pi)
    
    # Cache trig calculations
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Reuse sin_lat squared
    sin_lat_sq = sin_lat ** 2
    c = 1 / np.sqrt(1 + F * (F - 2) * sin_lat_sq)
    sq = c * (1 - F) ** 2

    achcp = (A * c + alt) * cos_lat
    x = achcp * cos_theta  # kilometers
    y = achcp * sin_theta
    z = (A * sq + alt) * sin_lat

    vx = -MFACTOR * y  # kilometers/second
    vy = MFACTOR * x
    vz = 0

    return (x, y, z), (vx, vy, vz)
