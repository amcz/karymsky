import volcat
import glob
import datetime
import os
import sys
import volcat
import xarray as xr
import numpy as np
import get_area
import hysplit

def get_volcat_1h(tdir, d1,d2):
# get all VOLCAT Files between two dates
    if os.path.exists(tdir):
        drange = None
        flist = glob.glob(tdir + '*nc')
        das = volcat.get_volcat_list(tdir,flist=None,verbose=True, daterange=[d1,d2],include_last=False)
        if len(das) == 0:
           print('NO VOLCAT FOUND')
           return [], None, None
        dset = volcat.combine_regridded(das)
        dset_averaged = dset.ash_mass_loading.mean(dim='time')
        dset_ht = dset.ash_cloud_height.max(dim='time')
        return das, dset_averaged, dset_ht



def volcat_mass(tdir,d1,d2):
    print(tdir)
    das, dset, dset_ht = get_volcat_1h(tdir,d1,d2)
    total_mass=[]
    
    for d in das:
        total_mass.append(float(d.ash_mass_loading_total_mass.values)) 

    print('Total mass volcat', np.mean(total_mass))
   # print('Max mass loading volcat', np.max(dset)) 
   # print(np.nanmin(dset)) 
    # another way which multiplies area by mass loading
    # field in data array must be in g/m2.
    # returns mass in Tg
   # print(volcat.check_total_mass(dset))
    return dset, dset_ht

def old(yset,d1,d2):
    vdir = '/hysplit3/alicec/projects/karymsky/volcat2/pc_corrected/'
    vmass, ht = volcat_mass(vdir,d1,d2)

    
    print(yset.time.values)
    mass = hysplit.hysp_massload(yset.HYSPLIT.sel(time=d1))
   
    area = get_area.get_area(mass.isel(source=0,ens=0))
    total = mass * area * 1e-6

    print('HYSPLIT total mass', np.nansum(total))
    print('HYSPLIT max mass loading', np.nanmax(mass.values))
    return vmass, mass


def cdf(clist, labels):
    from utilhysplit.evaluation import statmain
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    for c in zip(clist,labels):
        v = c[0].values
        v = v[~np.isnan(v)]
        v = v[v!=0]
        sdata,yval = statmain(cdf(v))
        statmain.plot_cdf(sdata,yval,ax)
    plt.xcale('log')  

def check1():
    ydir = '/hysplit3/alicec/projects/karymsky/HYSPLIT_results/'
    yfiles = glob.glob(ydir +'*nc')
    yset = xr.open_dataset(yfiles[2])
    d1 = datetime.datetime(2021,11,3,12)
    d2 = datetime.timedelta(hours=1) + d1
    vmass, hmass = main(yset,d1,d2) 
    return vmass,  hmass, yset



def check2():
    ydir = '/hysplit3/alicec/projects/karymsky/HYSPLIT_results/'
    yfiles = glob.glob(ydir +'*nc')
    yset = xr.open_dataset(yfiles[1])
    d1 = datetime.datetime(2021,11,3,12)
    d2 = datetime.timedelta(hours=1) + d1

