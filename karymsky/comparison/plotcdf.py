import datetime
import matplotlib.pyplot as plt
import numpy as np
from karymsky.io.readers import format_cdf_plot

def plot_cdf_multipanel(comparison, issue_date, minval=0.01, figsize=(15, 12)):
    """
    Create a multi-panel plot with CDF plots for each forecast date.
    
    Parameters:
    comparison (Comparison): The Comparison object from readers.py
    issue_date (datetime): The issue date of the forecast
    minval (float): Minimum value for mass loading (default: 0.01)
    figsize (tuple): Figure size (width, height) in inches
    
    Returns:
    fig, axes: The matplotlib figure and axes objects
    """
    # Ensure we have data for the issue date
    if issue_date not in comparison.datahash:
        comparison.get(issue_date)
    # Get forecast times starting from the issue date
    #forecast_times = comparison.forecast_times
    
    # Filter forecast times to only include those at or after the issue date
    #valid_forecast_times = [ft for ft in forecast_times if ft >= issue_date]
    dt = datetime.timedelta(hours=6)
    valid_forecast_times = []
    for a in [0,1,2,3]:
        valid_forecast_times.append(issue_date + dt*a)
    
       

 
    # Calculate the number of panels needed
    n_times = len(valid_forecast_times)
    print('HERE', n_times) 
    # Calculate optimal subplot layout
    if n_times <= 4:
        rows, cols = 2, 2
    elif n_times <= 6:
        rows, cols = 2, 3
    elif n_times <= 9:
        rows, cols = 3, 3
    elif n_times <= 12:
        rows, cols = 3, 4
    else:
        rows, cols = 4, 4
    
    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle the case where we have only one subplot
    if n_times == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    
    # Plot CDF for each forecast time
    for i, forecast_date in enumerate(valid_forecast_times):
        if i < len(axes):
            ax = axes[i]
            
            # Calculate time difference for title
            time_diff = forecast_date - issue_date
            hours_diff = int(time_diff.total_seconds() / 3600)
            
            # Create the CDF plot
            comparison.plot_mass_cdf(issue_date, forecast_date, minval=minval, ax=ax)
            
            # Format the plot
            title = f'T+{hours_diff}h ({forecast_date.strftime("%m-%d %H:%M")})'
            format_cdf_plot(ax, title=title)
    
    # Hide any unused subplots
    for i in range(n_times, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Mass Loading CDF - Issue Date: {issue_date.strftime("%Y-%m-%d %H:%M")}', 
                 fontsize=16, y=0.98)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig, axes

def plot_cdf_time_series(comparison, issue_date, specific_times=None, minval=0.01, figsize=(12, 8)):
    """
    Create a time series of CDF plots for specific forecast times.
    
    Parameters:
    comparison (Comparison): The Comparison object from readers.py
    issue_date (datetime): The issue date of the forecast
    specific_times (list): List of specific forecast times to plot (hours from issue_date)
                          If None, uses [3, 6, 12, 24] hours
    minval (float): Minimum value for mass loading (default: 0.01)
    figsize (tuple): Figure size (width, height) in inches
    
    Returns:
    fig, axes: The matplotlib figure and axes objects
    """
    if specific_times is None:
        specific_times = [3, 6, 12, 24]  # hours
    
    # Calculate forecast dates
    forecast_dates = [issue_date + datetime.timedelta(hours=h) for h in specific_times]
    
    # Create subplots
    n_plots = len(forecast_dates)
    cols = min(n_plots, 2)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if n_plots > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Ensure we have data for the issue date
    if issue_date not in comparison.datahash:
        comparison.get(issue_date)
    
    # Plot CDF for each specified time
    for i, (hours, forecast_date) in enumerate(zip(specific_times, forecast_dates)):
        if i < len(axes):
            ax = axes[i]
            
            # Create the CDF plot
            comparison.plot_mass_cdf(issue_date, forecast_date, minval=minval, ax=ax)
            
            # Format the plot
            title = f'T+{hours}h ({forecast_date.strftime("%m-%d %H:%M")})'
            format_cdf_plot(ax, title=title)
    
    # Hide any unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Mass Loading CDF Time Series - Issue Date: {issue_date.strftime("%Y-%m-%d %H:%M")}', 
                 fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig, axes

if __name__ == "__main__":
    # Example usage
    from karymsky.readers import Comparison
    
    # Example directories (adjust paths as needed)
    noaadir = "/hysplit3/alicec/projects/karymsky/HYSPLIT_results"
    metdir = "/hysplit3/alicec/projects/karymsky/MetOffice_results"
    bomdir = "/hysplit3/alicec/projects/karymsky/Bom"
    volcatdir = "/hysplit3/alicec/projects/karymsky/data"
    
    # Create comparison object
    comp = Comparison(noaadir, metdir, bomdir, volcatdir)
    
    # Example issue date
    issue_date = datetime.datetime(2021, 11, 3, 9)
    
    # Create multi-panel plot
    fig1, axes1 = plot_cdf_multipanel(comp, issue_date)
    plt.show()
    
    # Create time series plot for specific times
    fig2, axes2 = plot_cdf_time_series(comp, issue_date, specific_times=[3, 6, 12, 24])
    plt.show()
