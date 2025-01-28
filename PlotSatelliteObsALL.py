
'''
PlotSatellite.py

Program for plotting satellite observations. 

Loops through mutliple files with different output times. 
 
Options to vary:

   * Contour levels
   * Extent of axes
   * Lat and Long Grid lines
   * Ticks on ColorBar
'''

import iris
import numpy 
import iris.plot as iplt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mplcolors
import matplotlib
from matplotlib import colors, ticker
import glob
import datetime
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import obs_loader_ALL
import os
import datetime as dt
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm
import iris.plot as iplot
import sys


UTC_format = '%H%M%Z %d/%m/%Y'

def find_files(dir):
    ''' finds and orders the files in a directory
       returns a 2d list of file names and times for plume in datetime format
    '''

    file_list = []
    time_list = []

    for root,dirs,files in os.walk(dir):
        for name in files:
            #if 1==1:
            if 'va_out' in name:
                #strtime0 = name.split('_')[1] 
                #strtime = strtime0.split('.')[0] 
                #time=dt.datetime.strptime(strtime, "%Y%m%d%H%M")  		
                #file_list.append([os.path.join(root,name),time, strtime])
                file_list.append(os.path.join(root,name))

    file_list = [name for name in sorted(file_list,key=lambda one_file: one_file[1])]
    		
    return file_list

def setup_contours():      # ASH COLOUMN LOADING    

    colours = [[255, 255, 255],   #0 white
  	       [255, 219, 233],   #1 pale pink
  	       [255, 179, 255],   #2 pink
  	       [204, 153, 255],   #3
  	       [179, 170, 253],	  #4 
  	       [153, 153, 255],   #5
  	       [128, 170, 255],   #6
  	       [ 77, 210, 255],   #7
  	       [  0, 255, 255],   #8 cyan
  	       [  0, 232, 204],   #9
  	       [128, 255, 128],   #10 light green
	       [154, 225,   0],   #11 yellowgreen
  	       [204, 255,  51],   #12
  	       [255, 255,   0],   #13 yellow
  	       [255, 204,  36],   #14 orange
  	       [255, 153,  51],   #15
  	       [255, 102,   0],   #16
  	       [255,   0,   0],   #17 red
 	       [179,   0,   0],   #18 	       
      	       [154,   0,   0],   #19 
      	       #[128,    0,  0],  #20 maroon	   
      	       [196, 164, 132]]    #21 light brown - clear sky

    colours = numpy.array(colours)/255.    
    cmap = colors.ListedColormap(colours)
    levels = [ 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,2.75, 
               3.0,  3.5, 4.0,  4.5, 5.0, 7.5, 10.0, 100.0, 10e21] # 1e20 is clear sky
    pos_norm = BoundaryNorm(levels,21)
   
    return( cmap, levels, pos_norm)

def PlotSatellite(ashcube, clearskycube, date_object, time, PlotDir):


        plt.figure()

	# Set up axes
        ax = plt.axes(projection=ccrs.PlateCarree())

	# Set map extent

        ax.set_extent([140.0, 180.0, 45.0, 65.0])
        
	# Set up country outlines
        countries = cfeature.NaturalEarthFeature(
	    category='cultural',
	    name='admin_0_countries',
	    scale='50m',
	    facecolor='none')
        ax.add_feature(countries, edgecolor='black',zorder=2)

	# Set-up the gridlines
        gl = ax.gridlines(draw_labels=True, 
			  linewidth=0.8, 
			  alpha=0.9)
	
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

	# Plot
        contours = setup_contours()
        cmap=contours[0]
        norm=contours[2]

        #ashcube.data[ashcube.data <0.2] = numpy.nan
        ashplot = iplot.pcolormesh(ashcube,cmap=cmap,norm=norm,
                           edgecolors='None',rasterized=True)

        clearskycube.data[clearskycube.data <1e+20] = numpy.nan # NEED THIS OTHERWISE CLEARSKY CUBE DATA OF ZERO OVERLAPS ASH
        clearskyplot = iplot.pcolormesh(clearskycube,cmap=cmap,norm=norm,
                           edgecolors='None',rasterized=True)

       
    
        cb = plt.colorbar(ashplot, orientation='horizontal',shrink=0.9)#, ticks=	[0.5,1.0,1.5, 2.0,2.5,3.0,4.0,5.0,10.0])	
        # change colobar ticks labels and locators 
        cb.set_ticks([0.5,1.0,1.5, 2.0,2.5,3.0,4.0,5.0,10.0,10e21])
        cb.set_ticklabels(['0.5','1.0','1.5', '2.0','2.5','3.0','4.0','5.0','10.0','Clear \n sky'])
    
        cb.set_label('Ash column loading [gm-2]', fontsize = 10)
	
        #plot location of volcano
        plt.scatter(159.44, 54.05, s=80, c='black',edgecolors='black', linewidths=0.5, marker='^', alpha=0.8,transform=ccrs.PlateCarree()) #alpha: 0 (transparent) and 1 (opaque).

        #calculate total mass - within plotting domain
        ash=ashcube.extract(iris.Constraint(coord_values={'latitude':lambda cell: 45 < cell < 65, 'longitude':lambda cell: 140 < cell < 180} ))
        areas = iris.analysis.cartography.area_weights(ash) # calculate grid cell areas                                     
        totalmass_g=numpy.nansum(numpy.nansum(ash.data*areas))
        totalmass_kt = totalmass_g/1E9
        text4  = "Total mass = {:.2f}".format(totalmass_kt)

        # find maximum ash value and number of positive ash values and clear sky values
        text1  = "Maximum ash loading = {:.2f}".format(ash.data.max())
        text2 = "#PositiveAshObs = {:.0f}".format(len(ash.data[ash.data>0]))
        #text3 = "#ClearskyObs = {:.0f}".format(len(clearsky))

        #cube2 = ashcube.copy() 
        #cube2.data[cube2.data == 10e20] = 0.0

        
        date_object_num = dt.datetime.strptime(time, '%Y%m%d%H%M')
        date_object = str(date_object_num)  
        plt.title('Satellite observations \n'+date_object+' UTC \n' +text4+' kt \n'+ text1+' gm-2 \n' +text2, fontsize=10)

        output_filename=PlotDir+'HIM8_obs_all_available_'+time+'.png'

        plt.savefig(output_filename,dpi=150,bbox_inches='tight')
        plt.close()

        del ashcube




if __name__ == '__main__':

    PlotDir = '/data/users/apdg/INTEM/Karymsky_point25/PLOTS_obs_all_available/'
    # Create output directory if not existing 
    if not os.path.exists(PlotDir):   
        os.makedirs(PlotDir)

    Dir='/data/users/apdg/INTEM/Karymsky_point25/SatData/'

    file_list = find_files(Dir)

    for filename in file_list:
        #filename=filename[0]
        print(filename)
        if 'va_out' in filename:
            #print(filename)
            filesuffix = filename.rpartition(".")[0]
            #print(filesuffix)
            time = filesuffix.split('_')[2]
            #print(time)
            date_object_num = dt.datetime.strptime(time, '%Y%m%d%H%M')
            date_object = str(date_object_num)     
            #print(date_object)  
    
            ashcube = obs_loader_ALL.obs_to_cube(filename)      
            #print(ashcube)
            #print(numpy.max(ashcube.data))
            #print(numpy.min(ashcube.data))


            print('HIM8_'+time+'_cloudfree_out.csv')
            clearskycube = obs_loader_ALL.obs_to_cube(Dir+'HIM8_'+time+'_cloudfree_out.csv',clear_sky=True) # NEED TO ADD clear_sky=True
            #print(numpy.max(clearskycube.data))
            #print(numpy.min(clearskycube.data))
                  
            PlotSatellite(ashcube, clearskycube, date_object, time, PlotDir)   


