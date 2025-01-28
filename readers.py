import xarray as xr
import glob





class BaseNetCDF:

    def __init__(self,tdir):    
        self.tdir = tdir
        pass
   
    def get_files(self):
        ncfiles = glob.glob(self.tdir + '/*nc')
        return ncfiles 

    def forecast_times(self):
        d = datetime.datetime(2021,11,3,8)
        dlist = [d]
        dlist.append(datetime.datetime(2021,11,3,9))
        for iii in np.arange(0,24,3):
            dt = datetime.timedelta(hours=iii+1)
            dlist.append(d+dt)
        return dlist
