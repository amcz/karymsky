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
        dlist.append(datetime.datetime(2021, 11, 3, 12))
        #for iii in np.arange(3, 12, 3):
        #    dt = datetime.timedelta(hours=int(iii + 1))
        #    dlist.append(d + dt)
        d = dlist[-1]
        for iii in np.arange(6, 50, 6):
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
        for iii, ddd in enumerate(self.forecast_times()):
            #dstr = ddd.strftime("%Y%m%d%H%M")
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
        mass_4d = self.dset.HYSPLIT.sel(time=time)
       
        # finds closest lat 
        lats = self.dset.HYSPLIT.latitude.isel(x=0).values       
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
       
        # HYSPLIT output is in g/m3 return in mg/m3 
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
    ax.set_title(title)

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
    def forecast_times(self):
        return self.noaa.forecast_times()

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
        fig, axlist = map_util.setup_figure(fignum=1, rows=2, columns=2, central_longitude=180)
        axlist = axlist.flatten()
        # Use the same transform as the map projection
        transform = map_util.get_transform(central_longitude=0)  # Data coordinates are still 0-360 or -180/180
        
        # Calculate global bounds across all datasets
        all_masses = []
        for key in data.keys():
            mass = data[key].massload(fdate)
            mass = xr.where(mass < 0.01, np.nan, mass)
            all_masses.append(np.nanmax(mass.values))
        
        global_max = np.max(all_masses)
        if global_max > 10:
            bounds = [0.01, 0.2, 2, 5, 10, global_max]
        else:
            bounds = [0.01, 0.2, 2, 5, 9.90, 10]
        
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=5)
        cmap = plt.get_cmap('viridis', 5)
        
        # Store the pcolormesh objects for colorbar
        pcolormesh_obj = None
        
        for iii, key in enumerate(data.keys()):
            print(iii, key)
            mass = data[key].massload(fdate)
            mass = xr.where(mass < 0.01, np.nan, mass)
            
            if key == 'volcat':
                lat = mass.latitude.values
                lon = mass.longitude.values
            else:
                lat = data[key].latitude
                lon = data[key].longitude
            
            # Keep data coordinates as they are - don't transform longitude
            # The transform parameter handles the projection conversion
            pcolormesh_obj = axlist[iii].pcolormesh(lon, lat, mass.values, cmap=cmap, norm=norm, transform=transform)
            axlist[iii].plot(self.vloc[0],self.vloc[1],transform=transform,marker='^',color='r',markersize=10)
            # Place the text in the upper left corner
            lbl = key.upper()

            axlist[iii].text(0.05, 0.95, lbl , transform=axlist[iii].transAxes, fontsize=12, color='k', va='top')

            # Set extent based on forecast time and transformed coordinates
            if fdate == datetime.datetime(2021, 11, 3, 12):
                # For T+0, focus on source region
                axlist[iii].set_extent([150, 170, 50, 56], crs=transform)
            elif fdate > datetime.datetime(2021, 11, 3, 18):
                # For later times, show wider area as plume spreads
                axlist[iii].set_extent([160, 190, 48, 56], crs=transform)
            else:
                # Default extent for intermediate times
                axlist[iii].set_extent([150, 180, 48, 56], crs=transform)
            
            axlist[iii].set_xlabel('Longitude')
            map_util.format_plot(axlist[iii], transform, fsz=12)
        
        # Create a dedicated axis for the colorbar
        if pcolormesh_obj is not None:
            # Add space for colorbar on the right
            plt.subplots_adjust(right=0.85)
            # Create a new axis for the colorbar positioned to the right of the subplots
            cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
            fig.colorbar(pcolormesh_obj, cax=cbar_ax, label='Mass Loading (g m$^{-2}$)')
        
        axlist[0].set_title(f'{fdate.strftime("%Y-%m-%d %H:%M")}')
        
        return fig, axlist

    def plot_vertical_slices(self, date, time, latitude_target, figsize=(15, 10), 
                           min_conc=1e-6, log_scale=True, cmap='viridis'):
        """
        Plot vertical slices for all available datasets at a specific latitude.
        
        Parameters:
        date (datetime): Issue date of the forecast
        time (datetime): Forecast time
        latitude_target (float): Target latitude for the slice
        figsize (tuple): Figure size
        min_conc (float): Minimum concentration for plotting
        log_scale (bool): Use logarithmic color scale
        cmap (str): Colormap name
        
        Returns:
        tuple: (fig, axes_array)
        """
        if date not in self.datahash:
            self.get(date)
        
        data = self.datahash[date]
        available_datasets = [key for key in data.keys() if key != 'volcat']
        
        if not available_datasets:
            print("No datasets with vertical levels available")
            return None, None
        
        # Create subplots
        n_datasets = len(available_datasets)
        ncols = min(2, n_datasets)
        nrows = (n_datasets + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_datasets == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
       
        xmin=360
        xmax=-360
        ymax=100
        ymin=0 
        # Plot each dataset
        for i, key in enumerate(available_datasets):
            if i < len(axes):
                ax = axes[i]
                helper = data[key]
                
                try:
                    # Get vertical slice data
                    print(f"Getting vertical slice data for {key}...")
                    try:
                        lon, alt, conc_2d, actual_lat = helper.vertical_slice(time, latitude_target)
                        print(f"  {key}: Got data - lon shape: {lon.shape}, alt shape: {alt.shape}, conc_2d shape: {conc_2d.shape if conc_2d is not None else 'None'}")
                    except Exception as e:
                        print(f"  {key}: Error getting vertical slice data: {e}")
                        raise
                    
                    if conc_2d is not None:
                        cb = ax.pcolormesh(lon,alt,conc_2d)
                        print(f"  {key}: Setting labels and title...")
                        ax.set_xlabel('Longitude (°)', fontsize=12)
                        ax.set_ylabel('Altitude (m)', fontsize=12)
                        fig.colorbar(cb, ax=ax, label='Concentration (g/m³)', orientation='vertical')
                        #ax.set_title(f'{key.upper()}\nLat: {actual_lat:.2f}°N', fontsize=12)
                        #ax.grid(True, alpha=0.3)
                        #print(f"  {key}: Plot completed successfully")
      
                        bbox = get_vertical_slice_bounding_box(lon,alt,conc_2d,0.01)
                        if bbox['lon_min'] < xmin: xmin = bbox['lon_min']
                        if bbox['lon_max'] > xmax: xmax = bbox['lon_max']
                        if bbox['alt_max'] > ymax: ymax = bbox['alt_max']
                        #ax.set_xlim(bbox['lon_min'],bbox['lon_max'])
                        #ax.set_ylim(0,bbox['alt_max']+50) 
                    else:
                        print(f"  {key}: No data available")
                        ax.text(0.5, 0.5, f'No data available\nfor {key.upper()}', 
                               transform=ax.transAxes, ha='center', va='center')
                        ax.set_title(f'{key.upper()}\nNo data', fontsize=12)
                
                except Exception as e:
                    print(f"  {key}: Final error: {e}")
                    import traceback
                    traceback.print_exc()
                    ax.text(0.5, 0.5, f'Error plotting {key.upper()}:\n{str(e)[:50]}...', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{key.upper()}\nError', fontsize=12)
                ax.set_xlim(xmin,xmax)
                ax.set_ylim(ymin,ymax) 
                
        
        # Hide unused subplots
        #for i in range(n_datasets, len(axes)):
        #    axes[i].set_visible(False)
        
        #plt.suptitle(f'Vertical Cross-Sections at {latitude_target}°N\n{time.strftime("%Y-%m-%d %H:%M")}', 
        #             fontsize=14)
        #plt.tight_layout()
        return fig, axes

    def plot_vertical_profiles(self, date, time, longitude_target, latitude_target, 
                             figsize=(10, 8), ax=None):
        """
        Plot vertical profiles for all datasets at a specific location.
        
        Parameters:
        date (datetime): Issue date
        time (datetime): Forecast time
        longitude_target (float): Target longitude
        latitude_target (float): Target latitude
        figsize (tuple): Figure size
        ax (matplotlib.axes): Existing axes (optional)
        
        Returns:
        tuple: (fig, ax)
        """
        if date not in self.datahash:
            self.get(date)
        
        data = self.datahash[date]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        plotted_any = False
        
        for i, (key, helper) in enumerate(data.items()):
            if key != 'volcat':  # Skip volcat as it doesn't have vertical levels
                try:
                    alt, profile, actual_lon, actual_lat = helper.vertical_profile(
                        time, longitude_target, latitude_target
                    )
                    
                    if profile is not None and len(profile) > 0:
                        color = colors[i % len(colors)]
                        ax.plot(profile, alt, 'o-', label=key.upper(), 
                               linewidth=2, markersize=4, color=color)
                        plotted_any = True
                
                except Exception as e:
                    print(f"Could not plot profile for {key}: {e}")
        
        if plotted_any:
            ax.set_xlabel('Concentration (g/m³)', fontsize=12)
            ax.set_ylabel('Altitude (m)', fontsize=12)
            ax.set_title(f'Vertical Profiles\nLon: {longitude_target:.1f}°, Lat: {latitude_target:.1f}°\n'
                        f'{time.strftime("%Y-%m-%d %H:%M")}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data available for any dataset', 
                   transform=ax.transAxes, ha='center', va='center')
        
        return fig, ax

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

def get_vertical_slice_bounding_box(lon, alt, conc_2d, threshold):
    """
    Find bounding box around values above a given threshold in vertical slice data.
    
    Parameters:
    lon (array): Longitude coordinates (1D array)
    alt (array): Altitude coordinates (1D array) 
    conc_2d (array): 2D concentration data (lon x alt or alt x lon)
    threshold (float): Threshold value for concentration
    
    Returns:
    dict: Dictionary containing bounding box information:
        - 'lon_min': Minimum longitude with above-threshold values
        - 'lon_max': Maximum longitude with above-threshold values  
        - 'alt_min': Minimum altitude with above-threshold values
        - 'alt_max': Maximum altitude with above-threshold values
        - 'found': Boolean indicating if any above-threshold values were found
        - 'indices': Dictionary with 'lon_indices' and 'alt_indices' arrays
    """
    if conc_2d is None:
        return {
            'lon_min': None, 'lon_max': None,
            'alt_min': None, 'alt_max': None,
            'found': False, 'indices': None
        }
    
    # Ensure we have the right orientation - check if data matches coordinate dimensions
    if conc_2d.shape == (len(alt), len(lon)):
        # Data is (alt, lon) - transpose to (lon, alt) for consistency
        conc_2d = conc_2d.T
    elif conc_2d.shape != (len(lon), len(alt)):
        raise ValueError(f"Data shape {conc_2d.shape} doesn't match coordinate shapes lon:{len(lon)}, alt:{len(alt)}")
    
    # Find locations where concentration exceeds threshold
    above_threshold = conc_2d > threshold
    
    # Check if any values are above threshold
    if not np.any(above_threshold):
        return {
            'lon_min': None, 'lon_max': None,
            'alt_min': None, 'alt_max': None,
            'found': False, 'indices': None
        }
    
    # Get indices of above-threshold values
    lon_indices, alt_indices = np.where(above_threshold)
    
    # Find bounding box in index space
    lon_idx_min, lon_idx_max = np.min(lon_indices), np.max(lon_indices)
    alt_idx_min, alt_idx_max = np.min(alt_indices), np.max(alt_indices)
    
    # Convert to coordinate space
    lon_min, lon_max = lon[lon_idx_min], lon[lon_idx_max]
    alt_min, alt_max = alt[alt_idx_min], alt[alt_idx_max]
    
    return {
        'lon_min': lon_min,
        'lon_max': lon_max, 
        'alt_min': alt_min,
        'alt_max': alt_max,
        'found': True,
        'indices': {
            'lon_indices': lon_indices,
            'alt_indices': alt_indices,
            'lon_idx_min': lon_idx_min,
            'lon_idx_max': lon_idx_max,
            'alt_idx_min': alt_idx_min, 
            'alt_idx_max': alt_idx_max
        }
    }

def plot_vertical_slice_with_bbox(lon, alt, conc_2d, threshold, actual_lat=None, 
                                 title=None, ax=None, figsize=(12, 8), 
                                 log_scale=True, cmap='viridis'):
    """
    Plot vertical slice with bounding box overlay around above-threshold values.
    
    Parameters:
    lon (array): Longitude coordinates
    alt (array): Altitude coordinates
    conc_2d (array): 2D concentration data
    threshold (float): Threshold for bounding box
    actual_lat (float): Actual latitude of the slice (for title)
    title (str): Plot title (optional)
    ax (matplotlib.axes): Existing axes (optional)
    figsize (tuple): Figure size if creating new figure
    log_scale (bool): Use logarithmic color scale
    cmap (str): Colormap name
    
    Returns:
    tuple: (fig, ax, bbox_dict)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get bounding box
    bbox = get_vertical_slice_bounding_box(lon, alt, conc_2d, threshold)
    
    # Ensure proper data orientation for plotting
    if conc_2d.shape == (len(alt), len(lon)):
        conc_2d = conc_2d.T
    
    # Mask values below threshold for visualization
    min_conc = threshold * 0.1  # Show some values below threshold for context
    conc_masked = np.where(conc_2d > min_conc, conc_2d, np.nan)
    
    # Set up color scale
    if log_scale:
        try:
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=min_conc, vmax=np.nanmax(conc_masked))
        except:
            import matplotlib.colors as mcolors
            norm = mcolors.LogNorm(vmin=min_conc, vmax=np.nanmax(conc_masked))
    else:
        try:
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=min_conc, vmax=np.nanmax(conc_masked))
        except:
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=min_conc, vmax=np.nanmax(conc_masked))
    
    # Create the plot
    im = ax.pcolormesh(lon, alt, conc_2d, cmap=cmap, norm=norm, shading='nearest')
    
    # Add bounding box if values above threshold were found
    if bbox['found']:
        rect = patches.Rectangle(
            (bbox['lon_min'], bbox['alt_min']),
            bbox['lon_max'] - bbox['lon_min'],
            bbox['alt_max'] - bbox['alt_min'],
            linewidth=2, edgecolor='red', facecolor='none',
            label=f'Above {threshold:.1e}'
        )
        ax.add_patch(rect)
        ax.legend()
        
        # Add text with bounding box info
        info_text = (f"Threshold: {threshold:.1e}\n"
                    f"Lon: {bbox['lon_min']:.1f}° to {bbox['lon_max']:.1f}°\n"
                    f"Alt: {bbox['alt_min']:.0f} to {bbox['alt_max']:.0f} FL")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Concentration (mg/m³)', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Flight Level', fontsize=12)
    
    if title is None:
        if actual_lat is not None:
            title = f'Vertical Cross-Section at {actual_lat:.2f}°N'
        else:
            title = 'Vertical Cross-Section'
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    
    return fig, ax, bbox

    def plot_vertical_slices_with_bboxes(self, date, time, latitude_target, threshold,
                                       figsize=(15, 10), log_scale=True, cmap='viridis'):
        """
        Plot vertical slices with bounding boxes for all datasets.
        
        Parameters:
        date (datetime): Issue date of the forecast
        time (datetime): Forecast time
        latitude_target (float): Target latitude for the slice
        threshold (float): Concentration threshold for bounding boxes
        figsize (tuple): Figure size
        log_scale (bool): Use logarithmic color scale
        cmap (str): Colormap name
        
        Returns:
        tuple: (fig, axes, bbox_results)
        """
        if date not in self.datahash:
            self.get(date)
        
        data = self.datahash[date]
        available_datasets = [key for key in data.keys() if key != 'volcat']
        
        if not available_datasets:
            print("No datasets with vertical levels available")
            return None, None, None
        
        # Create subplots
        n_datasets = len(available_datasets)
        ncols = min(2, n_datasets)
        nrows = (n_datasets + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_datasets == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        bbox_results = {}
        
        # Plot each dataset with bounding box
        for i, key in enumerate(available_datasets):
            if i < len(axes):
                ax = axes[i]
                helper = data[key]
                
                try:
                    lon, alt, conc_2d, actual_lat = helper.vertical_slice(time, latitude_target)
                    
                    if conc_2d is not None:
                        # Plot with bounding box
                        fig_temp, ax_temp, bbox = plot_vertical_slice_with_bbox(
                            lon, alt, conc_2d, threshold, actual_lat,
                            title=f'{key.upper()}\nLat: {actual_lat:.2f}°N',
                            ax=ax, log_scale=log_scale, cmap=cmap
                        )
                        bbox_results[key] = bbox
                    else:
                        ax.text(0.5, 0.5, f'No data available\nfor {key.upper()}', 
                               transform=ax.transAxes, ha='center', va='center')
                        ax.set_title(f'{key.upper()}\nNo data', fontsize=12)
                        bbox_results[key] = {'found': False}
                
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error plotting {key.upper()}:\n{str(e)[:50]}...', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{key.upper()}\nError', fontsize=12)
                    bbox_results[key] = {'found': False, 'error': str(e)}
        
        # Hide unused subplots
        for i in range(n_datasets, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Vertical Cross-Sections with Bounding Boxes\n'
                    f'Lat: {latitude_target}°N, Threshold: {threshold:.1e}\n'
                    f'{time.strftime("%Y-%m-%d %H:%M")}', fontsize=14)
        plt.tight_layout()
        
        return fig, axes, bbox_results
