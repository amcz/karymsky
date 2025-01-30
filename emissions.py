import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from plotutils import colormaker
import plottcm


def process(df):
    columns = ['ht','top','lat','lon','width','date','duration','rate']
    df.columns = columns
    df['mass'] = df['rate']*3600
    df['psize'] = 1
    df['ht'] = df['ht']*1000
    df2 = df[['date','ht','mass','psize']]
    return df2

def get_met_emission_files():
    tdir = '/hysplit3/alicec/projects/karymsky/MetOffice_results/'
    fff = glob.glob(tdir + '*csv')
    return fff
    #df = pd.read_csv(f[10])
    #df2 = process(df)


# plot emissions from UK met office
def plot_met_emissions():
    fig = plt.figure(1,figsize=[15,5])
    ax = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    fff = get_met_emission_files()
    qqq = fff[0:8]
    cm = colormaker.ColorMaker('viridis',len(qqq),ctype='rgb')
    clrs = cm()


    for iii, f in enumerate(qqq):
        df = pd.read_csv(f)
        df2 = process(df)
        plottcm.plot_emissions_timeseries(df2,marker='.',log=True,ax=ax,clr=clrs[iii])
        plottcm.plot_emissions_profile(df2,marker='.',clr=clrs[iii],ax=ax2)

    for label in ax.get_xticklabels():
        label.set_rotation(45)




