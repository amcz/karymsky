"""
Emissions processing module for Karymsky volcano project.

This module provides functionality to process, retrieve and visualize volcanic emission data
from various sources including NOAA and UK Met Office. It includes functions to read emission
files, process the data into a standardized format, and create visualizations.
"""
#import datetime
import re
import glob
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
#from karymsky.plotutils import colormaker
#import plottcm


def sort_forecast_files(filenames,fkey='forecast'):
    import re
    def extract_forecast_number(filename):
        """Extract the forecast number from filename."""
        # Look for 'forecast' followed by one or more digits
        match = re.search(r'_(\d+)', filename)
        if match:
            return int(match.group(1))
        # If no match found, return a large number to put it at the end
        return 999

    # Sort filenames by the extracted forecast number
    sorted_files = sorted(filenames, key=extract_forecast_number)

    return sorted_files


def get_sorted_forecast_files(directory_pattern, forecast_pattern = '*forecast*'):
 # Combine directory and pattern
    if not directory_pattern.endswith('/'):
        directory_pattern += '/'

    full_pattern = directory_pattern + forecast_pattern

    # Get all matching files
    files = glob.glob(full_pattern)

    # Sort them numerically
    sorted_files = sort_forecast_files(files)

    return sorted_files



def process_noaa(df):
    """
    Process NOAA emission data into a standardized format.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing raw NOAA emission data
        
    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with standardized columns: date, ht, mass, psize
    """
    columns = ['date','mass','width','lat','lon','top','ht','duration','rate']
    df.columns=columns
    df['psize'] = 1
    df['ht'] = df['ht']*1000
    df['top'] = df['top']*1000
    df2 = df[['date','ht','top','mass','psize']]
    return df2 

def process(df):
    """
    Process UK Met Office emission data into a standardized format.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing raw UK Met Office emission data
        
    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with standardized columns: date, ht, mass, psize
    """
    columns = ['ht','top','lat','lon','width','date','duration','rate']
    df.columns = columns
    df['mass'] = df['rate']*3600
    df['psize'] = 1
    df['ht'] = df['ht']*1000
    df['top'] = df['top']*1000
    df2 = df[['date','ht','top','mass','psize']]
    return df2

def get_met_emission_files(tdir='/hysplit3/alicec/projects/karymsky/results/', version='m0'):
    """
    Get file paths for emission data based on version.
    
    Parameters
    ----------
    version : str, optional
        Version identifier determining which directory to use:
        - 'm1': MetOffice_results_Feb2025
        - 'm0': MetOffice_results restricts emission time period.
        - 'n0': HYSPLIT_results
        
    Returns
    -------
    list
        List of file paths to CSV emission files
    """
    if version=='m1':
        tdir = os.path.join(tdir, 'MetOffice_results_Feb2025/')
    elif version=='m0':
        tdir = os.path.join(tdir,'MetOffice_results/')
    elif version in ['n0','na']:
        tdir = os.path.join(tdir, 'HYSPLIT_results/')
        
    fff = glob.glob(tdir + '*csv')
    if version == 'n0':
        fff = [x for x in fff if 'forecast' in x]
    elif version == 'na':
        fff = [x for x in fff if 'apriori'  in x]
    return fff
    #df = pd.read_csv(f[10])
    #df2 = process(df)

def get_met_emissions(iii):
    """
    Get processed emission data from a specific file index.
    
    Parameters
    ----------
    iii : int
        Index of the file to retrieve from the sorted list of emission files
        
    Returns
    -------
    pandas.DataFrame
        Processed DataFrame containing emission data
    """
    fff = get_met_emission_files()
    df = pd.read_csv(fff[iii])
    df2 = process(df)
    return df2







