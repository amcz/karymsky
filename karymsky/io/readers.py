import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

import get_area
from karymsky.io.volcat_helper import volcat_mass
#from utilhysplit.evaluation import statmain
#from plotutils import map_util

"""

readers.py - read input netcdf files for inversion intercomparison

Author: Alice Crawford - NOAA Air Resources Laboratory

Changelog

2025 Feb 06 (amc) updated metoffice reader for new data.


"""



class BaseNetCDF:
    """
    Base class for reading NetCDF files.
    """

    def __init__(self, tdir):
        """
        Initialize the BaseNetCDF class.

        Parameters:
        tdir (str): Directory containing the NetCDF files.
        """
        if os.path.exists(tdir):
            self.tdir = tdir
        else:
            print('error cannot find directory')
        self.namehash = self.names()

    def get_forecast(self, date):
        """
        Get the forecast dataset for a specific date.

        Parameters:
        date (datetime): The date for which to get the forecast.

        Returns:
        xarray.Dataset: The forecast dataset.
        """
        fname = os.path.join(self.tdir, self.namehash[date])
        if os.path.exists(fname):
            return xr.open_dataset(fname)
        else:
            print('not found {}'.format(fname))

    def names(self):
        """
        Generate a dictionary of filenames based on forecast times.

        Returns:
        dict: Dictionary mapping dates to filenames.
        """
        namehash = {}
        for ddd in self.forecast_times():
            namehash[ddd] = ddd.strftime("%Y%m%d%H")
        return namehash

    def forecast_times(self):
        """
        Generate a list of forecast times.

        Returns:
        list: List of forecast times as datetime objects.
        """
        d = datetime.datetime(2021, 11, 3, 8)
        dlist = [d]
        dlist.append(datetime.datetime(2021, 11, 3, 9))
        dlist.append(datetime.datetime(2021, 11, 3, 12))
        #for iii in np.arange(3, 12, 3):
        #    dt = datetime.timedelta(hours=int(iii + 1))
        #    dlist.append(d + dt)
        d = dlist[-1]
        for iii in np.arange(6, 56, 6):
            dt = datetime.timedelta(hours=int(iii))
            dlist.append(d + dt)
            print(d+dt)
        return dlist

class MetOffice(BaseNetCDF):
    """
    Class for reading MetOffice NetCDF files.
    """

    def __init__(self, tdir):
        """
        Initialize the MetOffice class.

        Parameters:
        tdir (str): Directory containing the NetCDF files.
        """
        super().__init__(tdir)

    def get(self, date):
        """
        Get the MetOffice dataset for a specific date.

        Parameters:
        date (datetime): The date for which to get the dataset.

        Returns:
        MetOfficeHelper: Helper object for the dataset.
        """
        dset = self.get_forecast(date)
        return MetOfficeHelper(dset)

    def names(self):
        """
        Generate a dictionary of filenames based on forecast times.

        Returns:
        dict: Dictionary mapping dates to filenames.
        """
        namehash = {}
        for ddd in self.forecast_times():
            dstr = ddd.strftime("%Y%m%d%H%M")
            namehash[ddd] = f'NAME_InTEM_forecast_{dstr}.nc'
        return namehash

class Bom(BaseNetCDF):
    """
    Class for reading BOM NetCDF files.
    """

    def __init__(self, tdir):
        """
        Initialize the Bom class.

        Parameters:
        tdir (str): Directory containing the NetCDF files.
        """
        super().__init__(tdir)

    def get(self, date):
        """
        Get the BOM dataset for a specific date.

        Parameters:
        date (datetime): The date for which to get the dataset.

        Returns:
        BomHelper: Helper object for the dataset.
        """
        dset = self.get_forecast(date)
        return BomHelper(dset)

    def names(self):
        """
        Generate a dictionary of filenames based on forecast times.

        Returns:
        dict: Dictionary mapping dates to filenames.
        """
        namehash = {}
        for ddd in self.forecast_times():
            dstr = ddd.strftime("%Y%m%dT%H%MZ")
            namehash[ddd] = f'forecast_{dstr}.nc'
        return namehash

class NOAA(BaseNetCDF):
    """
    Class for reading NOAA NetCDF files.
    """

    def __init__(self, tdir):
        """
        Initialize the NOAA class.

        Parameters:
        tdir (str): Directory containing the NetCDF files.
        """
        self.set_type('apriori')
        super().__init__(tdir)

    def set_type(self,runtype):
        self.runtype=runtype 

    def get(self, date):
        """
        Get the NOAA dataset for a specific date.

        Parameters:
        date (datetime): The date for which to get the dataset.

        Returns:
        NoaaHelper: Helper object for the dataset.
        """
        dset = self.get_forecast(date)
        return NoaaHelper(dset)

    def names(self):
        """
        Generate a dictionary of filenames based on forecast times.

        Returns:
        dict: Dictionary mapping dates to filenames.
        """
        namehash = {}
        for iii, ddd in enumerate(self.forecast_times()):
            #dstr = ddd.strftime("%Y%m%d%H%M")
            if self.runtype=='apriori': 
                dstr = 'apriori_{}'.format(iii)
            else:
                dstr = 'forecast_{}'.format(iii)
            namehash[ddd] = f'HYSPLIT_inv_{dstr}.nc'
        return namehash

class DsetHelper:
    """
    Base helper class for datasets.
    """

    def __init__(self, dset):
        """
        Initialize the DsetHelper class.

        Parameters:
        dset (xarray.Dataset): The dataset to help with.
        """
        self.dset = dset

class NoaaHelper(DsetHelper):
    """
    Helper class for NOAA datasets.
    """

    @property
    def latitude(self):
        """
        Get the latitude values from the dataset.

        Returns:
        numpy.ndarray: Array of latitude values.
        """
        return self.dset.latitude.values

    @property
    def longitude(self):
        """
        Get the longitude values from the dataset.

        Returns:
        numpy.ndarray: Array of longitude values.
        """
        return self.dset.longitude.values

    def massload(self, time):
        """
        Calculate the mass loading for a specific time.

        Parameters:
        time (datetime): The time for which to calculate the mass loading.

        Returns:
        xarray.DataArray: The mass loading.
        """
        # check if dset is an xarray Dataset
        if not isinstance(self.dset, xr.Dataset):
            print('Dataset is not an xarray Dataset')
            return None
        if 'HYSPLIT' not in self.dset:
            print('HYSPLIT variable not found in dataset')
            return None
        timevalues = [pd.to_datetime(x) for x in self.dset.HYSPLIT.time.values]
        if time not in timevalues:
           print('NOAA time not available', time, timevalues)

        mass = self.dset.HYSPLIT.sel(time=time)
        # NOAA concentrations are in g/m3. 
        mass = mass * 1.524e3  # multiply by height of grid cell in meters.
        return mass.isel(ens=0, source=0).sum(dim='z')

    def vertical_slice(self, time, latitude_target, tolerance=0.5):
        """
        Extract a vertical cross-section (longitude vs altitude) at a specific latitude.
        
        Parameters:
        time (datetime): Time for which to extract data
        latitude_target (float): Target latitude for the slice
        tolerance (float): Tolerance for latitude matching (degrees)
        
        Returns:
        tuple: (longitude, altitude, concentration_2d, actual_latitude)
        """
        # Get 3D concentration data
        print('here') 
        mass_4d = self.dset.HYSPLIT.sel(time=time)
        
        # Find closest latitude
        lats = mass_4d.isel(x=0).latitude.values 
        #lats = self.latitude
        #if lats.ndim > 1:
        #    lats = lats[:, 0] if lats.shape[1] < lats.shape[0] else lats[0, :]
        print('here', type(lats), lats) 
        
        lat_diff = np.abs(lats - latitude_target)
        lat_idx = np.argmin(lat_diff)
        actual_lat = lats[lat_idx]
        
        if lat_diff[lat_idx] > tolerance:
            print(f"Warning: Closest latitude {actual_lat:.2f} is more than "
                  f"{tolerance} degrees from target {latitude_target:.2f}")
        
        # Get longitude and altitude
        lons = self.longitude
        if lons.ndim > 1:
            lons = lons[0, :] if lons.shape[0] < lons.shape[1] else lons[:, 0]
        
        alt = mass_4d.z.values
        # convert to FL
        alt = alt * 3.28084 / 100       
 
        # Extract vertical slice - NOAA uses 'y' dimension for latitude
        conc_slice = mass_4d.isel(ens=0, source=0, y=lat_idx)
       
        # hysplit output in g/m3 return in mg/m3 
        return lons, alt, conc_slice.values*1000, actual_lat

    def vertical_profile(self, time, longitude_target, latitude_target, tolerance=0.5):
        """
        Extract a vertical profile at a specific longitude and latitude.
        
        Parameters:
        time (datetime): Time for which to extract data
        longitude_target (float): Target longitude
        latitude_target (float): Target latitude
        tolerance (float): Tolerance for coordinate matching (degrees)
        
        Returns:
        tuple: (altitude, concentration_1d, actual_lon, actual_lat)
        """
        # Get 3D concentration data
        mass_4d = self.dset.HYSPLIT.sel(time=time)
        
        # Find closest coordinates
        lats = self.latitude
        lons = self.longitude
        
        if lats.ndim > 1:
            lats = lats[:, 0] if lats.shape[1] < lats.shape[0] else lats[0, :]
        if lons.ndim > 1:
            lons = lons[0, :] if lons.shape[0] < lons.shape[1] else lons[:, 0]
        
        lat_idx = np.argmin(np.abs(lats - latitude_target))
        lon_idx = np.argmin(np.abs(lons - longitude_target))
        
        actual_lat = lats[lat_idx]
        actual_lon = lons[lon_idx]
        
        # Get altitude and extract profile - NOAA uses 'y' and 'x' dimensions
        alt = mass_4d.z.values
        profile = mass_4d.isel(ens=0, source=0, y=lat_idx, x=lon_idx)
        
        return alt, profile.values, actual_lon, actual_lat

class BomHelper(DsetHelper):
    """
    Helper class for BOM datasets.
    """

    @property
    def latitude(self):
        """
        Get the latitude values from the dataset.

        Returns:
        numpy.ndarray: Array of latitude values.
        """
        return self.dset.latitude.values

    @property
    def longitude(self):
        """
        Get the longitude values from the dataset.

        Returns:
        numpy.ndarray: Array of longitude values.
        """
        return self.dset.longitude.values

    def convert_time(self, dtime):
        """
        Convert a datetime object to the corresponding index in the dataset.

        Parameters:
        dtime (datetime): The datetime object to convert.

        Returns:
        int: The index corresponding to the datetime object.
        """
        stamps = self.dset.timestamps.values
        stamps = [str(x) for x in stamps]
        d1 = datetime.datetime.isoformat(dtime)
        try:
            iii = stamps.index(d1)
        except:
            print('time not found in file', d1)
            print('times in file', stamps)
            iii = -1
        return iii

    def massload(self, time):
        """
        Calculate the mass loading for a specific time.
        BOM concentrations are in mg/m3.
        This method multiplies the concentration by the height of the grid cell in meters.
        but then must divide by 1000 to convert to g/m2

        Parameters:
        time (datetime): The time for which to calculate the mass loading.

        Returns:
        xarray.DataArray: The mass loading.
        """
        t = self.convert_time(time)
        if t >= 0:
            mass = self.dset.concentration.isel(time=t)
            mass = mass * 1.524  # multiply by height of grid cell in meters and divide by 1000 to convert to g/m2.
            return mass.sum(dim='levels')
        else:
            # Return None or empty array if time not found
            return None

    def vertical_slice(self, time, latitude_target, tolerance=0.5):
        """
        Extract a vertical cross-section (longitude vs altitude) at a specific latitude.
        
        Parameters:
        time (datetime): Time for which to extract data
        latitude_target (float): Target latitude for the slice
        tolerance (float): Tolerance for latitude matching (degrees)
        
        Returns:
        tuple: (longitude, altitude, concentration_2d, actual_latitude)
        """
        t_idx = self.convert_time(time)
        if t_idx < 0:
            return None, None, None, None
        
        # Get 3D concentration data
        conc_3d = self.dset.concentration.isel(time=t_idx)
        
        # Find closest latitude
        lats = self.latitude
        if lats.ndim > 1:
            lats = lats[:, 0] if lats.shape[1] < lats.shape[0] else lats[0, :]
        
        lat_diff = np.abs(lats - latitude_target)
        lat_idx = np.argmin(lat_diff)
        actual_lat = lats[lat_idx]
        
        if lat_diff[lat_idx] > tolerance:
            print(f"Warning: Closest latitude {actual_lat:.2f} is more than "
                  f"{tolerance} degrees from target {latitude_target:.2f}")
        
        # Get longitude and altitude
        lons = self.longitude
        if lons.ndim > 1:
            lons = lons[0, :] if lons.shape[0] < lons.shape[1] else lons[:, 0]
        
        alt = conc_3d.levels.values
        
        # Extract vertical slice
        conc_slice = conc_3d.isel(latitude=lat_idx)
        
        return lons, alt, conc_slice.values, actual_lat

    def vertical_profile(self, time, longitude_target, latitude_target, tolerance=0.5):
        """
        Extract a vertical profile at a specific longitude and latitude.
        
        Parameters:
        time (datetime): Time for which to extract data
        longitude_target (float): Target longitude
        latitude_target (float): Target latitude
        tolerance (float): Tolerance for coordinate matching (degrees)
        
        Returns:
        tuple: (altitude, concentration_1d, actual_lon, actual_lat)
        """
        t_idx = self.convert_time(time)
        if t_idx < 0:
            return None, None, None, None
        
        # Get 3D concentration data
        conc_3d = self.dset.concentration.isel(time=t_idx)
        
        # Find closest coordinates
        lats = self.latitude
        lons = self.longitude
        
        if lats.ndim > 1:
            lats = lats[:, 0] if lats.shape[1] < lats.shape[0] else lats[0, :]
        if lons.ndim > 1:
            lons = lons[0, :] if lons.shape[0] < lons.shape[1] else lons[:, 0]
        
        lat_idx = np.argmin(np.abs(lats - latitude_target))
        lon_idx = np.argmin(np.abs(lons - longitude_target))
        
        actual_lat = lats[lat_idx]
        actual_lon = lons[lon_idx]
        
        # Get altitude and extract profile
        alt = conc_3d.levels.values
        profile = conc_3d.isel(latitude=lat_idx, longitude=lon_idx)
        
        return alt, profile.values, actual_lon, actual_lat

class VolcatData:
    """
    Class for reading Volcat data.
    """

    def __init__(self, tdir):
        """
        Initialize the VolcatData class.

        Parameters:
        tdir (str): Directory containing the Volcat data.
        """
        self.tdir = tdir
        self.datahash = {}
        self.hthash = {}

    def reset(self):
        """
        Reset the data hash to empty.
        """
        self.datahash = {}

    def vertical_slice(self,time,latitude):
        ht = self.hthash[time]
        lats = ht.latitude.isel(x=0).values
        ldiff = np.abs(lats-latitude)
        idx = np.argmin(ldiff)
        htvalue = ht.isel(y=idx)  
        
        # Get longitude and height values
        lon_values = htvalue.longitude.values
        # Heights are in km. convert to FL. 3.28084*1000/100
        ht_values = htvalue.values * 32.8084
        
        # Filter out NaN height values
        valid_mask = ~np.isnan(ht_values)
        lon_filtered = lon_values[valid_mask]
        ht_filtered = ht_values[valid_mask]
        
        print(f"Original points: {len(lon_values)}, Valid points: {len(lon_filtered)}")
        return lon_filtered, ht_filtered

    
    def height(self,time):
        return self.hthash[time]     
 


    def get(self, date):
        """
        Get the Volcat data for a specific date.

        Parameters:
        date (datetime): The date for which to get the data.
        """
        d0 = date
        tlist = [0, 3, 6, 9, 12, 15, 18,24]
        #tlist = [0,3] 
        # Check if data for this date has already been loaded
        times_to_load = []
        for t in tlist:
            d1 = d0 + datetime.timedelta(hours=t)
            if d1 not in self.datahash:
                times_to_load.append((t, d1))
        
        # Only load data that hasn't been loaded yet
        for t, d1 in times_to_load:
            d2 = d1 + datetime.timedelta(hours=1)
            print('Looking for', d1, d2)
            mass, ht = volcat_mass(self.tdir, d1, d2)
            self.datahash[d1] = mass
            self.hthash[d1] = ht

    def massload(self, time):
        """
        Get the mass loading for a specific time.

        Parameters:
        time (datetime): The time for which to get the mass loading.

        Returns:
        xarray.DataArray: The mass loading.
        """
        return self.datahash[time]

def format_cdf_plot(ax, title=''):
    """
    Format the CDF plot.

    Parameters:
    ax (Axes): The Axes object to format.
    title (str): The title of the plot.
    """
    ax.set_xlabel('Mass Loading (g m$^{-2}$)')
    ax.set_ylabel('CDF')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_title(title)

class MetOfficeHelper(DsetHelper):
    """
    Helper class for MetOffice datasets.
    """

    @property
    def latitude(self):
        """
        Get the latitude values from the dataset.

        Returns:
        numpy.ndarray: Array of latitude values.
        """
        return self.dset.latitude

    @property
    def longitude(self):
        """
        Get the longitude values from the dataset.

        Returns:
        numpy.ndarray: Array of longitude values.
        """
        return self.dset.longitude

    def massload(self, time):
        """
        Calculate the mass loading for a specific time.

        Parameters:
        time (datetime): The time for which to calculate the mass loading.

        Returns:
        xarray.DataArray: The mass loading.
        """
        mass = self.concentration(time)
        mass = mass * 1.524  # multiply by height of grid cell in meters and divide by 1000 to convert to g/m2.
        return mass.sum(dim='flight_level')

    def concentration(self, time):
        """
        Get the concentration for a specific time.

        Parameters:
        time (datetime): The time for which to get the concentration.
                         beginning of averaging time.

        Returns:
        xarray.DataArray: The concentration.
        """
        # met office data gives hour ending.
        time = time + datetime.timedelta(hours=1)
        t = np.datetime64(time)
        tvals = self.dset.volcanic_ash_air_concentration.time.values
        if t not in tvals:
           print('Met office time not found in dset', t, tvals)
        try:
            conc = self.dset.volcanic_ash_air_concentration.sel(time=t)
        except:
            print('unable to retrieve conc')
            return False
        return conc

    def total_mass(self, time):
        """
        Calculate the total mass for a specific time.

        Parameters:
        time (datetime): The time for which to calculate the total mass.

        Returns:
        float: The total mass.
        """
        mass = self.massload(time)
        lat = self.latitude
        lon = self.longitude
        lon, lat = np.meshgrid(lon, lat)
        area = get_area.get_area2(lat, lon)
        total = area * 1e-6 * mass
        return np.nansum(total.values)

    def vertical_slice(self, time, latitude_target, tolerance=0.5):
        """
        Extract a vertical cross-section (longitude vs altitude) at a specific latitude.
        
        Parameters:
        time (datetime): Time for which to extract data
        latitude_target (float): Target latitude for the slice
        tolerance (float): Tolerance for latitude matching (degrees)
        
        Returns:
        tuple: (longitude, altitude, concentration_2d, actual_latitude)
        """
        # Get 3D concentration data
        conc_3d = self.concentration(time)
        
        # Find closest latitude
        lats = self.latitude
        if hasattr(lats, 'values'):
            lats = lats.values
        if lats.ndim > 1:
            lats = lats[:, 0] if lats.shape[1] < lats.shape[0] else lats[0, :]
        
        lat_diff = np.abs(lats - latitude_target)
        lat_idx = np.argmin(lat_diff)
        actual_lat = lats[lat_idx]
        
        if lat_diff[lat_idx] > tolerance:
            print(f"Warning: Closest latitude {actual_lat:.2f} is more than "
                  f"{tolerance} degrees from target {latitude_target:.2f}")
        
        # Get longitude and altitude
        lons = self.longitude
        if hasattr(lons, 'values'):
            lons = lons.values
        if lons.ndim > 1:
            lons = lons[0, :] if lons.shape[0] < lons.shape[1] else lons[:, 0]
        
        alt = conc_3d.flight_level.values
        
        # Extract vertical slice
        conc_slice = conc_3d.isel(latitude=lat_idx)
        
        return lons, alt, conc_slice.values, actual_lat

    def vertical_profile(self, time, longitude_target, latitude_target, tolerance=0.5):
        """
        Extract a vertical profile at a specific longitude and latitude.
        
        Parameters:
        time (datetime): Time for which to extract data
        longitude_target (float): Target longitude
        latitude_target (float): Target latitude
        tolerance (float): Tolerance for coordinate matching (degrees)
        
        Returns:
        tuple: (altitude, concentration_1d, actual_lon, actual_lat)
        """
        # Get 3D concentration data
        conc_3d = self.concentration(time)
        
        # Find closest coordinates
        lats = self.latitude
        lons = self.longitude
        
        if hasattr(lats, 'values'):
            lats = lats.values
        if hasattr(lons, 'values'):
            lons = lons.values
        
        if lats.ndim > 1:
            lats = lats[:, 0] if lats.shape[1] < lats.shape[0] else lats[0, :]
        if lons.ndim > 1:
            lons = lons[0, :] if lons.shape[0] < lons.shape[1] else lons[:, 0]
        
        lat_idx = np.argmin(np.abs(lats - latitude_target))
        lon_idx = np.argmin(np.abs(lons - longitude_target))
        
        actual_lat = lats[lat_idx]
        actual_lon = lons[lon_idx]
        
        # Get altitude and extract profile
        alt = conc_3d.flight_level.values
        profile = conc_3d.isel(latitude=lat_idx, longitude=lon_idx)
        
        return alt, profile.values, actual_lon, actual_lat
