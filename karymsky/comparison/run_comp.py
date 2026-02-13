import datetime
import matplotlib.pyplot as plt

from karymsky.comparison import comparison



class RunComp:

    def __init__(self):
        bdir = '/hysplit3/alicec/projects/karymsky/results/MetOffice_results_Feb2025/'
        adir = '/hysplit3/alicec/projects/karymsky/results/HYSPLIT_results/'
        cdir = '/hysplit3/alicec/projects/karymsky/results/Bom/'
        vdir = '/hysplit3/alicec/projects/karymsky/volcat2/pc_corrected2/'
        self.comp = comparison.Comparison(noaadir=adir,metdir=bdir,bomdir=cdir, volcatdir=vdir)
        self.ftimes = comp.noaa.forecast_times()
       
    def load(self):
        for f in self.ftimes:
            self.comp.get(f)


    def plots(self, ttt,fff,latitude_target=52):
        fget = self.ftimes[ttt]
        forecast_time = fget + datetime.timedelta(hours=fff)
        fig1, axlist1 = self.comp.plot_mass(fget,forcast_time)
        fig2, axlist2 = self.comp.plot_vertical_slices(fget,forecast_time,latitude_target=latitude_target) 
        plotcdf.plot_cdf_multipanel(comp,fget,minval=0.1)
