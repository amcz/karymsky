"""
Emissions processing module for Karymsky volcano project.

This module provides functionality to process, retrieve and visualize volcanic emission data
from various sources including NOAA and UK Met Office. It includes functions to read emission
files, process the data into a standardized format, and create visualizations.
"""
import datetime
import glob
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from plotutils import colormaker
import plottcm


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
    columns = ['date','mass','width','lat','lon','ht','top','duration','rate']
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

def get_met_emission_files(tdir='/hysplit3/alicec/projects/karymsky/', version='m0'):
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


def plot_emission_profiles(version='m0', nlist=None, edate=None,unit='mer'):
    fff = get_met_emission_files(version=version)
    fff.sort()
    if not nlist: nlist = np.arange(0,len(fff)+1)  
    cm = colormaker.ColorMaker('viridis',len(nlist)+1,ctype='rgb')
    clrs = cm()
    dlist = [datetime.datetime(2021,11,3,7)] 
    for i in range(1,6):
        dlist.append(dlist[-1] + datetime.timedelta(hours=1))
    n = len(dlist)
    fig, axs = plt.subplots(1, n, figsize=(5*n,10))
    ccc = 0
    for iii, f in enumerate(fff):
        if iii not in nlist: continue
        df = pd.read_csv(f)
        if 'm' in version:
            df2 = process(df)
        elif 'n' in version:
            df2 = process_noaa(df)
        for ddd, ax in zip(dlist, axs):
            dstr = ddd.strftime("%Y-%m-%d %H:%M:00")
            df3 = df2[df2['date'] == dstr]
            if not df3.empty:
                plottcm.plot_emissions_profile(df3,marker='.',clr=clrs[ccc],ax=ax, unit=unit)
        ccc+=1
    for ddd, ax in zip(dlist, axs):
        ax.set_title(f"{ddd}")
    axs[0].set_ylabel('altitude (km asl)',fontsize=15)



def time2label(ttt):
    if ttt=='202111030900': return 'forecast_1'
    if ttt=='202111031200': return 'forecast_2'
    if ttt=='202111031800': return 'forecast_3'
    if ttt=='202111040000': return 'forecast_4'
    if ttt=='202111040600': return 'forecast_5' 
    if ttt=='202111041200': return 'forecast_6' 
    if ttt=='202111041800': return 'forecast_7' 
    if ttt=='202111050000': return 'forecast_8' 
    if ttt=='202111050600': return 'forecast_9' 
    if ttt=='202111051200': return 'forecast_10' 
    if ttt=='202111051800': return 'forecast_11' 
    return ttt

# plot emissions from UK met office
def plot_met_emissions(version='m0',nlist=None, edate=None):
    """
    Plot emissions data from Met Office or NOAA.
    
    Creates a figure with two subplots:
    1. Time series of emissions
    2. Vertical profile of emissions
    
    Parameters
    ----------
    version : str, optional
        Version identifier determining which dataset to use:
        - 'm0'/'m1': UK Met Office data
        - 'n0': NOAA data
        Default is 'm0'
        
    Returns
    -------
    tuple
        (ax, ax2) - Matplotlib axes objects for the two subplots
    """
    fig = plt.figure(1,figsize=[15,5])
    ax = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    fff = get_met_emission_files(version=version)
    fff.sort()
    qqq = fff
    cm = colormaker.ColorMaker('viridis',len(qqq),ctype='rgb')
    clrs = cm()
    
    if not nlist: nlist = np.arange(0,len(fff)+1)  
    for iii, f in enumerate(qqq):
        if iii not in nlist: continue
        label = f.split('/')[-1]
        label = label.split('_')[-1]
        label = label.replace('.csv','')
        label = time2label(label)
        #label = datetime.datetime.strptime(label, '%Y%m%d%H%M')
        #label = label.strftime('%d %b %H:%M UTC')
        df = pd.read_csv(f)
        if 'm' in version:
            df2 = process(df)
        elif 'n' in version:
            df2 = process_noaa(df)
        if edate:
            print(df2['date'].unique())
            print(type(df2['date'].unique()[0]))
            df3 = df2[df2['date'] == edate]
        else:
            df3 = df2
        if not df3.empty:
            print(df3)
            plottcm.plot_emissions_timeseries(df2,marker='.',log=True,ax=ax,clr=clrs[iii],label=label)
            plottcm.plot_emissions_profile(df3,marker='.',clr=clrs[iii],ax=ax2)

    for label in ax.get_xticklabels():
        label.set_rotation(45)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower left', fontsize=10, ncol=2)
    return ax, ax2


