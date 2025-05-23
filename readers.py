import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import get_area
from compare_hysplit_volcat import volcat_mass
from utilhysplit.evaluation import statmain
from plotutils import map_util

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
        for iii in np.arange(3, 12, 3):
            dt = datetime.timedelta(hours=int(iii + 1))
            dlist.append(d + dt)
        d = dlist[-1]
        for iii in np.arange(6, 50, 6):
            dt = datetime.timedelta(hours=int(iii))
            dlist.append(d + dt)
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
        super().__init__(tdir)

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
        for ddd in self.forecast_times():
            dstr = ddd.strftime("%Y%m%d%H%M")
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
        mass = self.dset.HYSPLIT.sel(time=time)
        mass = mass * 1.524e3  # multiply by height of grid cell in meters.
        return mass.isel(ens=0, source=0).sum(dim='z')

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

        Parameters:
        time (datetime): The time for which to calculate the mass loading.

        Returns:
        xarray.DataArray: The mass loading.
        """
        t = self.convert_time(time)
        if t >= 0:
            mass = self.dset.concentration.isel(time=t)
        mass = mass * 1.524  # multiply by height of grid cell in meters.
        return mass.sum(dim='levels')

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
        mass = mass * 1.524  # multiply by height of grid cell in meters.
        return mass.sum(dim='flight_level')

    def concentration(self, time):
        """
        Get the concentration for a specific time.

        Parameters:
        time (datetime): The time for which to get the concentration.

        Returns:
        xarray.DataArray: The concentration.
        """
        time = time + datetime.timedelta(hours=1)
        t = np.datetime64(time)
        conc = self.dset.volcanic_ash_air_concentration.sel(time=t)
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

    def get(self, date):
        """
        Get the Volcat data for a specific date.

        Parameters:
        date (datetime): The date for which to get the data.
        """
        d0 = date
        tlist = [0, 3, 6, 9, 12, 15, 18, 24]
        for t in tlist:
            d1 = d0 + datetime.timedelta(hours=t)
            d2 = d1 + datetime.timedelta(hours=1)
            mass = volcat_mass(self.tdir, d1, d2)
            self.datahash[d1] = mass

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
    plt.title(title)

class Comparison:
    """
    Class for comparing different datasets.
    """

    def __init__(self, noaadir, metdir, bomdir, volcatdir):
        """
        Initialize the Comparison class.

        Parameters:
        noaadir (str): Directory containing NOAA data.
        metdir (str): Directory containing MetOffice data.
        bomdir (str): Directory containing BOM data.
        volcatdir (str): Directory containing Volcat data.
        """
        self.noaa = NOAA(noaadir)
        self.metoffice = MetOffice(metdir)
        self.bom = Bom(bomdir)
        self.volcat = VolcatData(volcatdir)
        self.datahash = {}

    @property   
    def vloc(self):
        return [159.4424,54.04855]

    @property
    def colorhash(self):
        """
        Get the color mapping for different datasets.

        Returns:
        dict: Dictionary mapping dataset names to colors.
        """
        clr = {}
        clr['noaa'] = 'b'
        clr['metoffice'] = 'g'
        clr['bom'] = 'c'
        clr['volcat'] = 'k'
        return clr

    def get(self, date):
        """
        Get the data for a specific date.

        Parameters:
        date (datetime): The date for which to get the data.
        """
        self.datahash[date] = {}
        self.datahash[date]['noaa'] = self.noaa.get(date)
        self.datahash[date]['metoffice'] = metoffice_data = self.metoffice.get(date)
        self.datahash[date]['bom'] = self.bom.get(date)
        self.volcat.get(date)
        self.datahash[date]['volcat'] = self.volcat

    def plot_mass_cdf(self, issue_date, forecast_date, minval=0.01, ax=None):
        """
        Plot the CDF of mass loading.

        Parameters:
        issue_date (datetime): The issue date of the forecast.
        forecast_date (datetime): The forecast date.
        minval (float): Minimum value for mass loading.
        ax (Axes): The Axes object to plot on.
        """
        if not ax:
            fig = plt.figure(1)
            ax = fig.add_subplot(1, 1, 1)
        data = self.datahash[issue_date]
        clrs = self.colorhash
        for key in data.keys():
            mass = data[key].massload(forecast_date)
            m = mass.values
            m = m[~np.isnan(m)]
            m = m[m > minval]
            sdata, yval = statmain.cdf(m)
            ax.step(sdata, yval, '-' + clrs[key], label=key)
        ax.set_xscale('log')

    def plot_mass(self, date, fdate):
        """
        Plot the mass loading for different datasets.

        Parameters:
        date (datetime): The date of the T+0 in the forecast
        fdate (datetime): The forecast date.
        """
        import matplotlib.colors as mcolors
        data = self.datahash[date]
        fig, axlist = map_util.setup_figure(fignum=1, rows=2, columns=2, central_longitude=0)
        axlist = axlist.flatten()
        transform = map_util.data_transform()
        norm = mcolors.LogNorm(vmin=0.01, vmax=100)
        norm = mcolors.Normalize(vmin=0.01, vmax=100)
        for iii, key in enumerate(data.keys()):
            mass = data[key].massload(fdate)
            mass = xr.where(mass < 0.01, np.nan, mass)
            massmax = np.nanmax(mass.values)
            if massmax > 10:
                bounds = [0.01, 0.2, 2, 5, 10, np.max(mass)]
            else:
                bounds = [0.01, 0.2, 2, 5, 9.90, 10]
            norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=5)
            cmap = plt.get_cmap('viridis', 5)
            if key == 'volcat':
                lat = mass.latitude.values
                lon = mass.longitude.values
            else:
                lat = data[key].latitude
                lon = data[key].longitude
            cb = axlist[iii].pcolormesh(lon, lat, mass.values, cmap=cmap, norm=norm, transform=transform)
            axlist[iii].plot(self.vloc[0],self.vloc[1],transform=transform,marker='^',color='r',markersize=10)
            if fdate == datetime.datetime(2021, 11, 3, 12):
                axlist[iii].set_xlim(150, 170)
                axlist[iii].set_ylim(50, 56)
            else:
                axlist[iii].set_xlim(150, 180)
                axlist[iii].set_ylim(48, 56)
            fig.colorbar(cb, ax=axlist[iii], label='Mass Loading')
            map_util.format_plot(axlist[iii], transform, fsz=12)
