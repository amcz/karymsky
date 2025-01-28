import datetime
import os
import numpy as np
import xarray as xr
import glob



class BaseNetCDF:

    def __init__(self,tdir): 
        if os.path.exists(tdir):   
            self.tdir = tdir
        else:
            print('error cannot find directory')
   
    def get_files(self):
        ncfiles = glob.glob(self.tdir + '/*nc')
        return ncfiles 

    def open_file(self,iii):
        nc = self.get_files()
        nc = nc[iii]
        dset = xr.open_dataset(nc)
        return dset   

    def names(self):
        namehash = {}
        for ddd in self.forecast_times():
            namehash[ddd] = ddd.strftime("%Y%m%d%H")
        return namehash 

    def forecast_times(self):
        d = datetime.datetime(2021,11,3,8)
        dlist = [d]
        dlist.append(datetime.datetime(2021,11,3,9))
        for iii in np.arange(3,12,3):
            dt = datetime.timedelta(hours=int(iii+1))
            dlist.append(d+dt)
        d = dlist[-1]
        for iii in np.arange(6,50,6):
            dt = datetime.timedelta(hours=int(iii))
            dlist.append(d+dt)
        return dlist

class MetOffice(BaseNetCDF):

    def __init__(self,tdir):
        super().__init__(tdir)    
        
    def open_file(self,iii):
        dset = super().open_file(iii) 
        return dset 

    def names(self):
        namehash = {}
        for ddd in self.forecast_times():
            dstr = ddd.strftime("%Y%m%d%H%M")
            namehash[ddd] = f'NAME_InTEM_forecast_{dstr}.nc'
        return namehash
    
    @staticmethod 
    def convert_time(dtime):
        dstr = dtime.strftime('%Y%m%d%H%M')
        return int(dstr) 

    def massload(self,dset,time):
        t = self.convert_time(time)
        mass = dset.concentration.sel(time=t)


