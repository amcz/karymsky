''' Load an observations file into an iris cube '''

import numpy as np
import datetime as dt
import sys
import iris
import iris.coords as icoords
import iris.coord_systems as icoord_systems
import iris.fileformats
import iris.io.format_picker as format_picker
import cf_units

def set_satgrid_size(filename):
    
    ''' The inversion observations file gives no details 
        on the lat-lon grid size so this is hardwired
    '''
    
    # All satellite data (MSG, MSG_global, HIM8 and GOES) use the same resolution
    dlat = 0.25 #  steps in latitude grid
    dlon = 0.25 # steps in longitude grid

    print(filename)   

    latmin = -90.0 # Corner lat of bottom left cell of output grid
    lonmin = -180.0 # corner lon of bottom left cell of output grid            
    nlat = 720 # Number of latitude cells
    nlon = 1440  # number of longitude cells
    
       
    lonmax = lonmin+(nlon*dlon)
    latmax = latmin+(nlat*dlat)
    
    print('Grid size: ')
    print('lonmin=', lonmin, 'lonmax=', lonmax, 'nlon=', nlon, 'dlon=', dlon)
    print('latmin=', latmin, 'latmax=', latmax, 'nlat=', nlat, 'dlat=', dlat)
    
  
    #calculate bounds
    lat1 = latmin + np.arange(nlat,dtype=np.float32)*dlat
    lat2 = latmin + dlat + np.arange(nlat,dtype=np.float32)*dlat
    latbound = [[l1,l2] for l1,l2 in zip(lat1,lat2)]
    lon1 = lonmin + np.arange(nlon,dtype=np.float32)*dlon
    lon2 = lonmin + dlon + np.arange(nlon,dtype=np.float32)*dlon
    lonbound = [[l1,l2] for l1,l2 in zip(lon1,lon2)]

    lat = latmin + dlat/2.0 + np.arange(nlat,dtype=np.float32)*dlat     
    lon = lonmin + dlon/2.0 + np.arange(nlon,dtype=np.float32)*dlon

#    lon_match = lon
#    for idx, value in enumerate(lon_match):
#        if value>180.0:
#	    lon_match[idx] = lon_match[idx]-360.0
    #print lon_match
    
    grid_size = [latmin, lonmin, latmax, lonmax, nlat, nlon, dlat, dlon, latbound, lonbound]

    return lat, lon, grid_size

def load_error(filename,clear_sky=False,obsfreq=3600,data_select='Mean_Ret_StDev'):
    
    ''' Load the observations file into a data array '''
        
    lat, lon, grid_size = set_satgrid_size(filename) 
    latmin = grid_size[0]
    lonmin = grid_size[1]
    latmax = grid_size[2]
    lonmax = grid_size[3]
    nlat = grid_size[4]    
    nlon = grid_size[5]    
    dlat = grid_size[6]    
    dlon = grid_size[7]
    latbound = grid_size[8]    
    lonbound = grid_size[9]    
    print('Extent:', lonmin, lonmax, latmin, latmax)

    
    
    #create the data array
    data = np.zeros((nlat,nlon),dtype=np.float32)
    
    #get time from filename
    #fname0 = filename.split('/')[-1]
    #print(fname0)
    #fname = fname0.split('_')[-1]
    #print(fname)
    #print(fname.split('.csv')[0])
    filesuffix = filename.rpartition(".")[0]
    time = filesuffix.split('_')[1]   
    filetime = dt.datetime.strptime(time,'%Y%m%d%H%M')
    time_coord = [filetime - dt.timedelta(seconds=3600),filetime]
    
    #build the coordinate dictionary
    coords = {'latitude':[lat,0],'longitude':[lon,1],'time':time_coord}
    bounds = {'latitude':latbound,'longitude':lonbound}
        
    #read file
    file_handler = open(filename,'r')
    headers = next(file_handler).strip().split(',')
    headers = [header.strip() for header in headers]
    
    for i,line in enumerate(file_handler):
        vals = line.split(',')
        
        #check time matches file time and observation freqency is correct
        start_ind = headers.index('T start')
        end_ind = headers.index('T end')
        
        start = dt.datetime.strptime(vals[start_ind].strip(),'%Y/%m/%d %H:%M:%S')
        end = dt.datetime.strptime(vals[end_ind].strip(),'%Y/%m/%d %H:%M:%S')
        
        freq = (end -start).total_seconds()
        
        if freq != obsfreq:
            print('incorrect observations frequency on file line', i+2)
            print('frequency is :', freq, 'seconds')
            print('required frequenct is :', obsfreq, 'seconds')
            sys.exit()
        
        if end != filetime:
            print('time for observations on line', i+2, 
                  'does not match the time given in the file name')
            sys.exit()
            
        #get latitude and longitude for observation
        clonmin = float(vals[headers.index('X_Lon1')])
        clonmax = float(vals[headers.index('X_Lon2')])
        clatmin = float(vals[headers.index('Y_Lat1')])
        clatmax = float(vals[headers.index('Y_Lat2')])
        
        lat_coord = clatmin + (clatmax - clatmin)/2.0
        lon_coord = clonmin + (clonmax - clonmin)/2.0

        #find index of observation in data array
        lat_ind = np.where(lat==lat_coord)[0][0]
	#if lon_coord<0.0:
	#    lon_coord = lon_coord + 360.0
        lon_ind = np.where(lon==lon_coord)[0][0]
	
        
        #get observation and scale to g/m2 if clear sky set obs value large
        if clear_sky:
            obs = 10e20
        else:
            area = cell_area(clatmin,clatmax,clonmin,clonmax)
            
            obs = float(vals[headers.index(data_select)])/area

        # set 0.0 to clear sky and a value of 10e-10
        #if obs==0.0:
        #    obs=10e20
        #assign observation to the data array
        data[lat_ind,lon_ind] = obs
	
            
    file_handler.close()

    return bounds, coords, data



#def load_obs(filename,clear_sky=False,obsfreq=3600,data_select='Ash_conc_median'):
def load_obs(filename,clear_sky=False,obsfreq=3600,data_select='Ash_conc_mean'):
    
    ''' Load the observations file into a data array '''
        
    lat, lon, grid_size = set_satgrid_size(filename) 
    latmin = grid_size[0]
    lonmin = grid_size[1]
    latmax = grid_size[2]
    lonmax = grid_size[3]
    nlat = grid_size[4]    
    nlon = grid_size[5]    
    dlat = grid_size[6]    
    dlon = grid_size[7]
    latbound = grid_size[8]    
    lonbound = grid_size[9]    
    print('Extent:', lonmin, lonmax, latmin, latmax)

    
    
    #create the data array
    data = np.zeros((nlat,nlon),dtype=np.float32)
    
    #get time from filename
    #fname0 = filename.split('/')[-1]
    #print(fname0)
    #fname = fname0.split('_')[-1]
    #print(fname)
    #print(fname.split('.csv')[0])
    filesuffix = filename.rpartition(".")[0]
    time = filesuffix.split('_')[2]   
    filetime = dt.datetime.strptime(time,'%Y%m%d%H%M')
    time_coord = [filetime - dt.timedelta(seconds=3600),filetime]
    
    #build the coordinate dictionary
    coords = {'latitude':[lat,0],'longitude':[lon,1],'time':time_coord}
    bounds = {'latitude':latbound,'longitude':lonbound}
        
    #read file
    file_handler = open(filename,'r')
    headers = next(file_handler).strip().split(',')
    headers = [header.strip() for header in headers]
    
    for i,line in enumerate(file_handler):
        vals = line.split(',')
        
        #check time matches file time and observation freqency is correct
        start_ind = headers.index('T start')
        end_ind = headers.index('T end')
        
        start = dt.datetime.strptime(vals[start_ind].strip(),'%Y/%m/%d %H:%M:%S')
        end = dt.datetime.strptime(vals[end_ind].strip(),'%Y/%m/%d %H:%M:%S')
        
        freq = (end -start).total_seconds()
        
        if freq != obsfreq:
            print('incorrect observations frequency on file line', i+2)
            print('frequency is :', freq, 'seconds')
            print('required frequenct is :', obsfreq, 'seconds')
            sys.exit()
        
        if end != filetime:
            print('time for observations on line', i+2, 
                  'does not match the time given in the file name')
            sys.exit()
            
        #get latitude and longitude for observation
        clonmin = float(vals[headers.index('X_Lon1')])
        clonmax = float(vals[headers.index('X_Lon2')])
        clatmin = float(vals[headers.index('Y_Lat1')])
        clatmax = float(vals[headers.index('Y_Lat2')])
        
        lat_coord = clatmin + (clatmax - clatmin)/2.0
        lon_coord = clonmin + (clonmax - clonmin)/2.0

        #find index of observation in data array
        lat_ind = np.where(lat==lat_coord)[0][0]
	#if lon_coord<0.0:
	#    lon_coord = lon_coord + 360.0
        lon_ind = np.where(lon==lon_coord)[0][0]
	
        
        #get observation and scale to g/m2 if clear sky set obs value large
        if clear_sky:
            obs = 10e20
        else:
            area = cell_area(clatmin,clatmax,clonmin,clonmax)
            
            if data_select == 'low_error':
            
                #obs = float(vals[headers.index('Ash_conc_median')])/area
                obs = float(vals[headers.index('Ash_conc_mean')])/area
                obs = obs - float(vals[headers.index('Mean_Ret_StDev')])/area
               
            elif data_select == 'high_error':
            
                #obs = float(vals[headers.index('Ash_conc_median')])/area
                obs = float(vals[headers.index('Ash_conc_mean')])/area
                obs = obs + float(vals[headers.index('Mean_Ret_StDev')])/area       
                
            else:
                obs = float(vals[headers.index(data_select)])/area

        # set 0.0 to clear sky and a value of 10e-10
        #if obs==0.0:
        #    obs=10e20
        #assign observation to the data array
        data[lat_ind,lon_ind] = obs
	
            
    file_handler.close()

    return bounds, coords, data
        

def obserror_to_cube(filename,clear_sky=False,obsfreq=3600,data_select='Mean_Ret_StDev'):

    ''' Loads the observation data array into an iris cube'''

    bounds, coords, data = load_error(filename,clear_sky,obsfreq,data_select)
    
    #create a cube
    cube = iris.cube.Cube(data)

    #rename the cube
    cube.rename('OBSERVATION ERRORS')

    #units are g/m2
    cube.units = 'g/m2'

    #define the time unit
    time_unit = cf_units.Unit('hours since epoch', calendar='gregorian')

    #coordinate system
    lat_lon_coord_system = icoord_systems.GeogCS(6371229)

    #build time, latitude and longitude coordinates
    for key,coord in coords.items():
        if key == 'latitude' or key == 'longitude':
            coord_units = 'degrees'
            coord_sys = lat_lon_coord_system
            icoord = icoords.DimCoord(points=coord[0],
                                      bounds=bounds[key],
                                      standard_name=key,
                                      units=coord_units,
                                      coord_system=coord_sys)
            cube.add_dim_coord(icoord,coord[1])
                                      
        if key == 'time':
            coord_units = time_unit
            pts = time_unit.date2num(coord[1])
            icoord = icoords.AuxCoord(points=pts,
                                      standard_name=key,
                                      units=coord_units,
                                      coord_system=None)

            bnds = time_unit.date2num(np.vstack((coord[0],coord[1])).T)
            icoord.bounds = bnds
            cube.add_aux_coord(icoord)
                
    return cube

#def obs_to_cube(filename,clear_sky=False,obsfreq=3600,data_select='Ash_conc_median'):
def obs_to_cube(filename,clear_sky=False,obsfreq=3600,data_select='Ash_conc_mean'):

    ''' Loads the observation data array into an iris cube'''

    bounds, coords, data = load_obs(filename,clear_sky,obsfreq,data_select)
    
    #create a cube
    cube = iris.cube.Cube(data)

    #rename the cube
    cube.rename('OBSERVATIONS')

    #units are g/m2
    cube.units = 'g/m2'

    #define the time unit
    time_unit = cf_units.Unit('hours since epoch', calendar='gregorian')

    #coordinate system
    lat_lon_coord_system = icoord_systems.GeogCS(6371229)

    #build time, latitude and longitude coordinates
    for key,coord in coords.items():
        if key == 'latitude' or key == 'longitude':
            coord_units = 'degrees'
            coord_sys = lat_lon_coord_system
            icoord = icoords.DimCoord(points=coord[0],
                                      bounds=bounds[key],
                                      standard_name=key,
                                      units=coord_units,
                                      coord_system=coord_sys)
            cube.add_dim_coord(icoord,coord[1])
                                      
        if key == 'time':
            coord_units = time_unit
            pts = time_unit.date2num(coord[1])
            icoord = icoords.AuxCoord(points=pts,
                                      standard_name=key,
                                      units=coord_units,
                                      coord_system=None)

            bnds = time_unit.date2num(np.vstack((coord[0],coord[1])).T)
            icoord.bounds = bnds
            cube.add_aux_coord(icoord)
                
    return cube

def cell_area(latmin,latmax,lonmin,lonmax):
    
    ''' convert an observation in g to g/m2 '''
    
    R = 6367300.0     #radius of earth
    n = 1000          #segments for more accurate latitude
    dtor = 0.0174533  #degress to radians
    
    #split area into nxn cells
    dlon = (lonmax - lonmin)/float(n)
    dlat = (latmax - latmin)/float(n)
    
    #arrange n into np array to avoid using loops
    all_n = np.arange(n,dtype=float)
    
    # calculate area for one longitude
    dlatmet = R*dlat*dtor
    lat = latmin + dlat/2.0 + dlat*all_n
    dlonmet = R*dlon*dtor*np.cos(lat*dtor)
    area = dlonmet*dlatmet
    
    # multiply by n for all longitude
    area = area.sum()*float(n)
    
    return area
    
    
    
