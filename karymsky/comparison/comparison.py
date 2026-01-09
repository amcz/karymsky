import datetime
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from utilhysplit.evaluation import statmain
from plotutils import map_util
from karymsky.io.readers import NOAA, MetOffice, Bom, VolcatData

"""
comparison.py - Compare different volcanic ash datasets

Author: Alice Crawford - NOAA Air Resources Laboratory

Extracted from readers.py for better code organization.

23 Dec 2026 (amc) started fixing vertical slice plot

"""


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
        return [159.4424, 54.04855]

    @property
    def colorhash(self):
        """
        Get the color mapping for different datasets.

        Returns:
        dict: Dictionary mapping dataset names to colors.
        """
        clr = {}
        clr["noaa"] = "b"
        clr["metoffice"] = "g"
        clr["bom"] = "c"
        clr["volcat"] = "k"
        return clr

    def reset(self):
        """
        Reset the data hash.
        """
        self.datahash = {}

    def get(self, date):
        """
        Get the data for a specific date.

        Parameters:
        date (datetime): The date for which to get the data.
        """
        # Check if data for this date has already been loaded
        if date in self.datahash:
            print(f"Data for {date} already loaded, skipping...")
            return

        self.datahash[date] = {}
        self.datahash[date]["noaa"] = self.noaa.get(date)
        self.datahash[date]["metoffice"] = metoffice_data = self.metoffice.get(date)
        self.datahash[date]["bom"] = self.bom.get(date)
        self.volcat.get(date)
        self.datahash[date]["volcat"] = self.volcat

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
            ax.step(sdata, yval, "-" + clrs[key], label=key)
        ax.set_xscale("log")

    def get_extents(self, fdate):
        if fdate <= datetime.datetime(2021, 11, 3, 12):
            extent = [150, 170, 50, 56]
        elif fdate <= datetime.datetime(2021, 11, 3, 18):
            extent = [150, 180, 48, 56]
        elif fdate <= datetime.datetime(2021, 11, 4, 0):
            extent = [155, 185, 48, 56]
        elif fdate <= datetime.datetime(2021, 11, 4, 6):
            extent = [160, 190, 48, 56]
        elif fdate <= datetime.datetime(2021, 11, 4, 12):
            extent = [160, 195, 46, 56]
        elif fdate <= datetime.datetime(2021, 11, 4, 18):
            extent = [160, 200, 46, 58]
        elif fdate <= datetime.datetime(2021, 11, 5, 0):
            extent = [160, 205, 44, 58]
        elif fdate <= datetime.datetime(2021, 11, 5, 6):
            extent = [160, 210, 44, 60]
        elif fdate <= datetime.datetime(2021, 11, 5, 12):
            extent = [160, 215, 42, 60]
        elif fdate <= datetime.datetime(2021, 11, 5, 18):
            extent = [160, 220, 42, 62]
        elif fdate <= datetime.datetime(2021, 11, 20, 18):
            extent = [160, 220, 42, 62]
        else:
            extent = [150, 180, 48, 56]
        return extent

    

    def plot_mass(self, date, fdate):
        """
        Plot the mass loading for different datasets.

        Parameters:
        date (datetime): The date of the T+0 in the forecast
        fdate (datetime): The forecast date.
        """
        import matplotlib.colors as mcolors

        data = self.datahash[date]
        fig, axlist = map_util.setup_figure(
            fignum=1, rows=2, columns=2, central_longitude=180
        )
        axlist = axlist.flatten()
        # Use the same transform as the map projection
        transform = map_util.get_transform(
            central_longitude=0
        )  # Data coordinates are still 0-360 or -180/180

        # Calculate global bounds across all datasets
        all_masses = []
        keylist = data.keys()
        keylist = [x for x in keylist if "volcat" not in x]
        for key in keylist:
            # print(list(data[key].massload.keys()))
            mass = data[key].massload(fdate)
            mass = xr.where(mass < 0.01, np.nan, mass)
            all_masses.append(np.nanmax(mass.values))
        # vmass = data['volcat'].datahash[fdate]
        # all_masses.append(np.nanmax(vmass.values))

        global_max = np.max(all_masses)
        if global_max > 10:
            bounds = [0.01, 0.05, 0.2, 2, 5, 10, global_max]
        else:
            bounds = [0.01, 0.05, 0.2, 2, 5, 9.90, 10]

        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
        cmap = plt.get_cmap("viridis", 5)

        # Store the pcolormesh objects for colorbar
        pcolormesh_obj = None
        keylist.append("volcat")

        for iii, key in enumerate(keylist):
            try:
                mass = data[key].massload(fdate)
                mass = xr.where(mass < 0.01, np.nan, mass)

                if key == "volcat":
                    lat = mass.latitude.values
                    lon = mass.longitude.values
                else:
                    lat = data[key].latitude
                    lon = data[key].longitude

                # Keep data coordinates as they are - don't transform longitude
                # The transform parameter handles the projection conversion
                pcolormesh_obj = axlist[iii].pcolormesh(
                    lon, lat, mass.values, cmap=cmap, norm=norm, transform=transform
                )
                axlist[iii].plot(
                    self.vloc[0],
                    self.vloc[1],
                    transform=transform,
                    marker="^",
                    color="r",
                    markersize=10,
                )
                # Place the text in the upper left corner
                lbl = key.upper()

                axlist[iii].text(
                    0.05,
                    0.95,
                    lbl,
                    transform=axlist[iii].transAxes,
                    fontsize=12,
                    color="k",
                    va="top",
                )

                # Set extent based on forecast time and transformed coordinates
                if fdate <= datetime.datetime(2021, 11, 3, 12):
                    # T+0: Focus on source region
                    axlist[iii].set_extent([150, 170, 50, 56], crs=transform)
                elif fdate <= datetime.datetime(2021, 11, 3, 18):
                    # T+6: Plume begins to spread eastward
                    axlist[iii].set_extent([150, 180, 48, 56], crs=transform)
                elif fdate <= datetime.datetime(2021, 11, 4, 0):
                    # T+12: Wider eastward spread
                    axlist[iii].set_extent([155, 185, 48, 56], crs=transform)
                elif fdate <= datetime.datetime(2021, 11, 4, 6):
                    # T+18: Further eastward movement
                    axlist[iii].set_extent([160, 190, 48, 56], crs=transform)
                elif fdate <= datetime.datetime(2021, 11, 4, 12):
                    # T+24: Continued eastward spread
                    axlist[iii].set_extent([160, 195, 46, 56], crs=transform)
                elif fdate <= datetime.datetime(2021, 11, 4, 18):
                    # T+30: Maximum eastward extent
                    axlist[iii].set_extent([160, 200, 46, 58], crs=transform)
                elif fdate <= datetime.datetime(2021, 11, 5, 0):
                    # T+36: Far field dispersion
                    axlist[iii].set_extent([160, 205, 44, 58], crs=transform)
                elif fdate <= datetime.datetime(2021, 11, 5, 6):
                    # T+42: Extended far field
                    axlist[iii].set_extent([160, 210, 44, 60], crs=transform)
                elif fdate <= datetime.datetime(2021, 11, 5, 12):
                    # T+48: Very far field
                    axlist[iii].set_extent([160, 215, 42, 60], crs=transform)
                elif fdate <= datetime.datetime(2021, 11, 5, 18):
                    # T+54: Maximum extent
                    axlist[iii].set_extent([160, 220, 42, 62], crs=transform)
                elif fdate <= datetime.datetime(2021, 11, 20, 18):
                    # T+54: Maximum extent
                    axlist[iii].set_extent([160, 220, 42, 62], crs=transform)
                else:
                    # Default extent for any other times
                    axlist[iii].set_extent([150, 180, 48, 56], crs=transform)

                axlist[iii].set_xlabel("Longitude")
                map_util.format_plot(axlist[iii], transform, fsz=12)

            except Exception as e:
                # Handle errors for individual datasets (especially volcat)
                print(f"Error plotting {key}: {e}")
                lbl = key.upper()
                axlist[iii].text(
                    0.5,
                    0.5,
                    f"{lbl}\nError: Data not available",
                    transform=axlist[iii].transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                axlist[iii].text(
                    0.05,
                    0.95,
                    lbl,
                    transform=axlist[iii].transAxes,
                    fontsize=12,
                    color="k",
                    va="top",
                )
                # Set same extent as other plots for consistency
                if fdate == datetime.datetime(2021, 11, 3, 12):
                    # T+0: Focus on source region
                    axlist[iii].set_extent([150, 170, 50, 56], crs=transform)
                elif fdate == datetime.datetime(2021, 11, 3, 18):
                    # T+6: Plume begins to spread eastward
                    axlist[iii].set_extent([150, 180, 48, 56], crs=transform)
                elif fdate == datetime.datetime(2021, 11, 4, 0):
                    # T+12: Wider eastward spread
                    axlist[iii].set_extent([155, 185, 48, 56], crs=transform)
                elif fdate == datetime.datetime(2021, 11, 4, 6):
                    # T+18: Further eastward movement
                    axlist[iii].set_extent([160, 190, 48, 56], crs=transform)
                elif fdate == datetime.datetime(2021, 11, 4, 12):
                    # T+24: Continued eastward spread
                    axlist[iii].set_extent([165, 195, 46, 56], crs=transform)
                elif fdate == datetime.datetime(2021, 11, 4, 18):
                    # T+30: Maximum eastward extent
                    axlist[iii].set_extent([170, 200, 46, 58], crs=transform)
                elif fdate == datetime.datetime(2021, 11, 5, 0):
                    # T+36: Far field dispersion
                    axlist[iii].set_extent([175, 205, 44, 58], crs=transform)
                elif fdate == datetime.datetime(2021, 11, 5, 6):
                    # T+42: Extended far field
                    axlist[iii].set_extent([180, 210, 44, 60], crs=transform)
                elif fdate == datetime.datetime(2021, 11, 5, 12):
                    # T+48: Very far field
                    axlist[iii].set_extent([185, 215, 42, 60], crs=transform)
                elif fdate == datetime.datetime(2021, 11, 5, 18):
                    # T+54: Maximum extent
                    axlist[iii].set_extent([190, 220, 42, 62], crs=transform)
                else:
                    # Default extent for any other times
                    axlist[iii].set_extent([150, 180, 48, 56], crs=transform)
                map_util.format_plot(axlist[iii], transform, fsz=12)

        # Create a dedicated axis for the colorbar
        if pcolormesh_obj is not None:
            # Add space for colorbar on the right
            plt.subplots_adjust(right=0.85)
            # Create a new axis for the colorbar positioned to the right of the subplots
            cbar_ax = fig.add_axes(
                [0.87, 0.15, 0.03, 0.7]
            )  # [left, bottom, width, height]
            fig.colorbar(pcolormesh_obj, cax=cbar_ax, label="Mass Loading (g m$^{-2}$)")

        axlist[0].set_title(f'{fdate.strftime("%Y-%m-%d %H:%M")}')

        return fig, axlist

    def plot_vertical_slices(
        self,
        date,
        time,
        latitude_target,
        figsize=(15, 10),
        min_conc=1e-6,
        log_scale=True,
        cmap="viridis",
    ):
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
        available_datasets = [key for key in data.keys() if key != "volcat"]

        if not available_datasets:
            print("No datasets with vertical levels available")
            return None, None

        # Create subplots
        n_datasets = len(available_datasets)
        ncols = min(2, n_datasets)
        nrows = (n_datasets + ncols - 1) // ncols
        nrows = 1
        ncols = 3

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_datasets == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()

        extent = self.get_extents(time)

        vhelper = data['volcat']
        vslice = vhelper.vertical_slice(time, latitude_target)
        # Plot each dataset
        for i, key in enumerate(available_datasets):
            if i < len(axes):
                ax = axes[i]
                helper = data[key]

                try:
                    # Get vertical slice data
                    lon, alt, conc_2d, actual_lat = helper.vertical_slice(
                        time, latitude_target
                    )
                except:
                    print("problem with vertical slice for", key)
                # ----------------------------------------------------------------------------------------------
                try:
                    if conc_2d is not None:
                        print("creating meshgrid")
                        # Create meshgrid for plotting
                        lon_mesh, alt_mesh = np.meshgrid(lon, alt)

                        # Mask values below minimum concentration
                        # conc_masked = np.where(conc_2d.T > min_conc, conc_2d.T, np.nan)
                        conc_masked = np.where(conc_2d > min_conc, conc_2d, np.nan)

                        # Set up color scale
                        if log_scale:
                            from matplotlib.colors import LogNorm

                            norm = LogNorm(vmin=min_conc, vmax=np.nanmax(conc_masked))
                        else:
                            norm = plt.Normalize(
                                vmin=min_conc, vmax=np.nanmax(conc_masked)
                            )

                except Exception as eee:
                    print("problem with plotting for", key, eee)
                    # Create the plot
                # ----------------------------------------------------------------------------------------------
                try:
                    im = ax.pcolormesh(lon, alt, conc_masked, cmap=cmap)

                except Exception as eee:
                    print("problem with pcolormesh", eee)
                ax.plot(vslice[0], vslice[1], color='r', linestyle='--', label='Volcat Slice')
                # ----------------------------------------------------------------------------------------------
                try:
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label("Concentration (mg/m³)", fontsize=10)

                    # Set labels and title
                    ax.set_xlabel("Longitude (°)", fontsize=12)
                    ax.set_ylabel("Altitude (m)", fontsize=12)
                    ax.set_title(f"{key.upper()}\nLat: {actual_lat:.2f}°N", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    # ax.text(0.5, 0.5, f'No data available\nfor {key.upper()}',
                    #       transform=ax.transAxes, ha='center', va='center')
                    # ax.set_title(f'{key.upper()}\ndata', fontsize=12)

                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error plotting {key.upper()}:\n{str(e)[:50]}...",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                    ax.set_title(f"{key.upper()}\nError", fontsize=12)
                # ----------------------------------------------------------------------------------------------
                ax.set_xlim([extent[0], extent[1]])
                ax.set_ylim([0,600])
        # Hide unused subplots
        for i in range(n_datasets, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(
            f'Vertical Cross-Sections at {latitude_target}°N\n{time.strftime("%Y-%m-%d %H:%M")}',
            fontsize=14,
        )
        plt.tight_layout()
        return fig, axes

    def plot_vertical_profiles(
        self, date, time, longitude_target, latitude_target, figsize=(10, 8), ax=None
    ):
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

        colors = ["blue", "green", "red", "orange", "purple"]
        plotted_any = False

        for i, (key, helper) in enumerate(data.items()):
            if key != "volcat":  # Skip volcat as it doesn't have vertical levels
                try:
                    alt, profile, actual_lon, actual_lat = helper.vertical_profile(
                        time, longitude_target, latitude_target
                    )

                    if profile is not None and len(profile) > 0:
                        color = colors[i % len(colors)]
                        ax.plot(
                            profile,
                            alt,
                            "o-",
                            label=key.upper(),
                            linewidth=2,
                            markersize=4,
                            color=color,
                        )
                        plotted_any = True

                except Exception as e:
                    print(f"Could not plot profile for {key}: {e}")

        if plotted_any:
            ax.set_xlabel("Concentration (g/m³)", fontsize=12)
            ax.set_ylabel("Altitude (m)", fontsize=12)
            ax.set_title(
                f"Vertical Profiles\nLon: {longitude_target:.1f}°, Lat: {latitude_target:.1f}°\n"
                f'{time.strftime("%Y-%m-%d %H:%M")}',
                fontsize=14,
            )
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No data available for any dataset",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        return fig, ax
