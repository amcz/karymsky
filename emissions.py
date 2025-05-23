import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from plotutils import colormaker
import plottcm

def process_noaa(df):
    columns = ['date','mass','width','lat','lon','ht','top','duration','rate']
    df.columns=columns
    df['psize'] = 1
    df['ht'] = df['ht']*1000
    df2 = df[['date','ht','mass','psize']]
    return df2 

def process(df):
    columns = ['ht','top','lat','lon','width','date','duration','rate']
    df.columns = columns
    df['mass'] = df['rate']*3600
    df['psize'] = 1
    df['ht'] = df['ht']*1000
    df2 = df[['date','ht','mass','psize']]
    return df2

def get_met_emission_files(version='m1'):
    if version=='m1':
        tdir = '/hysplit3/alicec/projects/karymsky/MetOffice_results_Feb2025/'
    elif version=='m0':
        tdir = '/hysplit3/alicec/projects/karymsky/MetOffice_results/'
    elif version=='n0':
        tdir = '/hysplit3/alicec/projects/karymsky/HYSPLIT_results/'
        
    fff = glob.glob(tdir + '*csv')
    return fff
    #df = pd.read_csv(f[10])
    #df2 = process(df)

def get_met_emissions(iii):
    fff = get_met_emission_files()
    df = pd.read_csv(fff[iii])
    df2 = process(df)
    return df2
    


# plot emissions from UK met office
def plot_met_emissions(version='m0'):
    fig = plt.figure(1,figsize=[15,5])
    ax = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    fff = get_met_emission_files(version=version)
    fff.sort()
    if 'm' in version:
        qqq = fff[1:8]
    else:
        qqq = fff
    cm = colormaker.ColorMaker('viridis',len(qqq),ctype='rgb')
    clrs = cm()


    for iii, f in enumerate(qqq):
        print(iii, f)
        df = pd.read_csv(f)
        if 'm' in version:
            df2 = process(df)
        elif 'n' in version:
            df2 = process_noaa(df)

        plottcm.plot_emissions_timeseries(df2,marker='.',log=True,ax=ax,clr=clrs[iii])
        plottcm.plot_emissions_profile(df2,marker='.',clr=clrs[iii],ax=ax2)

    for label in ax.get_xticklabels():
        label.set_rotation(45)

    return ax, ax2


