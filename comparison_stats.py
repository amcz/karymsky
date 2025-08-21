"""
Script to read in Met Office, NOAA and BoM inversion forecast data and compare against MO and Volcat satellite data.
"""

import glob
import iris
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import xarray as xr
import os, sys
import datetime
import volcat
import numpy
import obs_loader_ALL

UTC_format = '%H:%M UTC %d/%m/%Y'

def Model_Obs_compare(M_dir, B_dir, N_dir, MO_sat_dir, VOLCAT_sat_dir, times, outdir, threshold, threshold_volcat):

    MO_FMS_Full=[]
    BoM_FMS_Full=[]
    NOAA_FMS_Full=[]
    MO_FMS_Volcat_Full=[]
    BoM_FMS_Volcat_Full=[]
    NOAA_FMS_Volcat_Full=[]

    MO_precision_Full=[]
    BoM_precision_Full=[]
    NOAA_precision_Full=[]
    MO_precision_Volcat_Full=[]
    BoM_precision_Volcat_Full=[]
    NOAA_precision_Volcat_Full=[]
    
    MO_pod_Full=[]
    BoM_pod_Full=[]
    NOAA_pod_Full=[]
    MO_pod_Volcat_Full=[]
    BoM_pod_Volcat_Full=[]
    NOAA_pod_Volcat_Full=[]
    
    MO_pofd_Full=[]
    BoM_pofd_Full=[]
    NOAA_pofd_Full=[]
    MO_pofd_Volcat_Full=[]
    BoM_pofd_Volcat_Full=[]
    NOAA_pofd_Volcat_Full=[]
    
    MO_pofa_Full=[]
    BoM_pofa_Full=[]
    NOAA_pofa_Full=[]
    MO_pofa_Volcat_Full=[]
    BoM_pofa_Volcat_Full=[]
    NOAA_pofa_Volcat_Full=[]

    n_MOsat_Full=[]
    n_Volcat_Full=[]
    timestr_Full=[]
    plot_time_Full=[]
    
    
# Loop over issue times
    for timestep in times:
        
#Read MO model data
        MTotCol = read_MO_data(M_dir, timestep)
        
#Read BoM model data
        BTotCol = read_BoM_data(B_dir, timestep)
                
# Convert timestep string to date-time
#        print(timestep)

        yyyy = int(timestep[:4])
        mm = int(timestep[4:6])
        dd = int(timestep[6:8])
        hh = int(timestep[8:10])
        minmin = int(timestep[10:])
            
        t0 = datetime.datetime(yyyy, mm, dd, hh, minmin)
            
#        print(t0)

# Read NOAA model data
        NTotCol = read_NOAA_data(N_dir, t0)
        
#        print(len(MTotCol.coord('time').points))
#        print(len(BTotCol.coord('time').points))
        
# Checks on data - cubes have same number of time dimensions
        if len(MTotCol.coord('time').points) != len(BTotCol.coord('time').points):
            print('MO and BoM time arrays are different')
        else:
            print('MO and BoM time arrays are the same size')
        
        hour_start_str=[]
        plot_time=[]
        hour_end_str=[]
        n_sat=[]
        n_sat_limit=[]
        n_MO_limit=[]
        n_BoM_limit=[]
        n_NOAA_limit=[]
        n_volcat=[]
        n_volcat_limit=[]
        n_MO_volcat_limit=[]
        n_BoM_volcat_limit=[]
        n_NOAA_volcat_limit=[]

        MO_FMS = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        BoM_FMS = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        NOAA_FMS = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        MO_FMS_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        BoM_FMS_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        NOAA_FMS_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 

        MO_precision = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        BoM_precision = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        NOAA_precision = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        MO_precision_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        BoM_precision_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        NOAA_precision_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 

        MO_pod = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        BoM_pod = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        NOAA_pod = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        MO_pod_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        BoM_pod_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        NOAA_pod_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 

        MO_pofd = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        BoM_pofd = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        NOAA_pofd = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        MO_pofd_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        BoM_pofd_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        NOAA_pofd_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 

        MO_pofa = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        BoM_pofa = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        NOAA_pofa = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        MO_pofa_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        BoM_pofa_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        NOAA_pofa_VOLCAT = numpy.empty(len(MTotCol.coord('time').points), dtype=object) 
        
#        print(f'timestep: {timestep}')
#        print('NOAA times')
#        print(NTotCol.coord('time'))
#        print(NTotCol.coord('time').points)
#        print(NTotCol.coord('time').units.num2pydate(NTotCol.coord('time').points))
#        print(NTotCol.coord('time').units.num2pydate(NTotCol.coord('time').points)[3])
#        print('MO times')
#        print(MTotCol.coord('time'))
#        print(MTotCol.coord('time').units.num2pydate(MTotCol.coord('time').points))
#        print(MTotCol.coord('time').units.num2pydate(MTotCol.coord('time').points)[0])
#        
#        print('BoM times')
#        print(BTotCol.coord('time'))
#        if NTotCol.coord('time').units.num2pydate(NTotCol.coord('time').points)[3] == MTotCol.coord('time').units.num2pydate(MTotCol.coord('time').points)[0]:
#            print('NOAA and MO times match')
#        else:
#            print('No Match - NOAA and MO times')

# Loop over forecast times within an issue time
        for i in range(len(MTotCol.coord('time').points)):
            
            addhourstart = i * 3
            hour_start = t0 + datetime.timedelta(hours=addhourstart)
            hour_end = hour_start + datetime.timedelta(hours=1)
            
# Find corresponding NOAA time (Note MO time is labelled hour ending, whereas NOAA is labelled hour start)
            NOAA_index = -999
            for j in range(len(NTotCol.coord('time').points)):
                
# Take one hour off Met Office valid time to get NOAA valid time
                NOAA_time= MTotCol.coord('time').units.num2pydate(MTotCol.coord('time').points)[i] + datetime.timedelta(hours=-1)
            
                if NTotCol.coord('time').units.num2pydate(NTotCol.coord('time').points)[j] == NOAA_time:
                    NOAA_index = j
                    break
# Check corresponding NOAA time found
            if NOAA_index < 0:
                print('NOAA index not found')
                print(f'timestep: {timestep}')
                Temp_time=MTotCol.coord('time').units.num2pydate(MTotCol.coord('time').points)[i]
                print(f'forecast time: {Temp_time}')
                exit()
            print(f'timestep: {timestep}')
            print(f'hour start: {hour_start.strftime(UTC_format)}')
            print('MO index: ',i)
            print(MTotCol.coord('time').units.num2pydate(MTotCol.coord('time').points)[i])
            print('NOAA index: ',NOAA_index)
            print(NTotCol.coord('time').units.num2pydate(NTotCol.coord('time').points)[NOAA_index])
            
            timestart_str = hour_start.strftime('%Y%m%d%H%M')
            
            
            hour_start_str.append(hour_start.strftime(UTC_format))
            plot_time.append(hour_start)
            hour_end_str.append(hour_end.strftime(UTC_format))

# Read in Volcat data            
            VOLCAT_da = read_volcat_data(VOLCAT_sat_dir, hour_start, hour_end)
            
#            print('VOLCAT data')
#            print(VOLCAT_da)
#            print(VOLCAT_da.data)
#            print(VOLCAT_da.data.shape)
#            print(VOLCAT_da.data.max())
#            print(VOLCAT_da.data.min())
#            print('Volcat Coordinates')
#            print(VOLCAT_da.coords['latitude'][:,0])
#            print(VOLCAT_da.coords['longitude'][0,:])
            
# Read in MO satellite data            
            MO_sat_ash, MO_sat_clear = read_MO_sat_data(MO_sat_dir, hour_end)
            
#            print('NOAA_index: ',NOAA_index)
#            print(NTotCol)
#            print(NTotCol[:,:,NOAA_index])
            
#Screen data using only gridpoints for which all model data and observations exist
            
# MO satellite data
            lonarr, latarr, satarr, Marr, Barr, Narr = subselect_MO(MTotCol[i], BTotCol[i], NTotCol[:,:,NOAA_index], MO_sat_ash, MO_sat_clear, t0)
            
# Write arrays out to files
            write_data_file(lonarr, latarr, satarr, Marr, Barr, Narr, timestep, timestart_str, outdir)

            
#Calculate statistics from MO satellite data
#            print('Calculation statistics using MO satellite data')

# Using MO model data
#            print('MO model data')
            MO_FMS[i], MO_precision[i], MO_pod[i], MO_pofd[i], MO_pofa[i], n_sat_points, n_sat_threshold, n_MO_threshold = calc_stats(satarr, Marr, threshold)

# Using BoM model data
#            print('BoM model data')
            BoM_FMS[i], BoM_precision[i], BoM_pod[i], BoM_pofd[i], BoM_pofa[i], n_sat_points, n_sat_threshold, n_BoM_threshold = calc_stats(satarr, Barr, threshold)

# Using NOAA model data
#            print('MO model data')
            NOAA_FMS[i], NOAA_precision[i], NOAA_pod[i], NOAA_pofd[i], NOAA_pofa[i], n_sat_points, n_sat_threshold, n_NOAA_threshold = calc_stats(satarr, Narr, threshold)
            
            n_sat.append(n_sat_points)
            n_sat_limit.append(n_sat_threshold)
            n_MO_limit.append(n_MO_threshold)
            n_BoM_limit.append(n_BoM_threshold)
            n_NOAA_limit.append(n_NOAA_threshold)

# Screen data using only gridpoints for which all model data and observations exist. 
# Note data for which all model and satellite data is zero is removed.
            
            # Volcat satellite data

            lonarr_volcat, latarr_volcat, Volcatarr, Marr_volcat, Barr_volcat, Narr_volcat = subselect_Volcat(MTotCol[i], BTotCol[i], NTotCol[:,:,NOAA_index], VOLCAT_da, t0)

            # Write arrays out to files
            write_data_file(lonarr_volcat, latarr_volcat, Volcatarr, Marr_volcat, Barr_volcat, Narr_volcat, timestep, timestart_str, outdir, True)

            #Calculate statistics from VOLCAT satellite data
#            print('Calculation statistics using VOLCAT satellite data')

            # Using MO model data
#            print('MO model data')
            MO_FMS_VOLCAT[i], MO_precision_VOLCAT[i], MO_pod_VOLCAT[i], MO_pofd_VOLCAT[i], MO_pofa_VOLCAT[i], n_volcat_points, n_volcat_threshold, n_MO_VOLCAT_threshold = calc_stats(Volcatarr, Marr_volcat, threshold_volcat)

            # Using BoM model data
#            print('BoM model data')
            BoM_FMS_VOLCAT[i], BoM_precision_VOLCAT[i], BoM_pod_VOLCAT[i], BoM_pofd_VOLCAT[i], BoM_pofa_VOLCAT[i], n_volcat_points, n_volcat_threshold, n_BoM_VOLCAT_threshold = calc_stats(Volcatarr, Barr_volcat, threshold_volcat)

            # Using NOAA model data
#            print('MO model data')
            NOAA_FMS_VOLCAT[i], NOAA_precision_VOLCAT[i], NOAA_pod_VOLCAT[i], NOAA_pofd_VOLCAT[i], NOAA_pofa_VOLCAT[i], n_volcat_points, n_volcat_threshold, n_NOAA_VOLCAT_threshold = calc_stats(Volcatarr, Narr_volcat, threshold_volcat)

            n_volcat.append(n_volcat_points)
            n_volcat_limit.append(n_volcat_threshold)
            n_MO_volcat_limit.append(n_MO_VOLCAT_threshold)
            n_BoM_volcat_limit.append(n_BoM_VOLCAT_threshold)
            n_NOAA_volcat_limit.append(n_NOAA_VOLCAT_threshold)
            
        MO_FMS_Full.append(MO_FMS)
        BoM_FMS_Full.append(BoM_FMS)
        NOAA_FMS_Full.append(NOAA_FMS)
        MO_FMS_Volcat_Full.append(MO_FMS_VOLCAT)
        BoM_FMS_Volcat_Full.append(BoM_FMS_VOLCAT)
        NOAA_FMS_Volcat_Full.append(NOAA_FMS_VOLCAT)

        MO_precision_Full.append(MO_precision)
        BoM_precision_Full.append(BoM_precision)
        NOAA_precision_Full.append(NOAA_precision)
        MO_precision_Volcat_Full.append(MO_precision_VOLCAT)
        BoM_precision_Volcat_Full.append(BoM_precision_VOLCAT)
        NOAA_precision_Volcat_Full.append(NOAA_precision_VOLCAT)

        MO_pod_Full.append(MO_pod)
        BoM_pod_Full.append(BoM_pod)
        NOAA_pod_Full.append(NOAA_pod)
        MO_pod_Volcat_Full.append(MO_pod_VOLCAT)
        BoM_pod_Volcat_Full.append(BoM_pod_VOLCAT)
        NOAA_pod_Volcat_Full.append(NOAA_pod_VOLCAT)

        MO_pofd_Full.append(MO_pofd)
        BoM_pofd_Full.append(BoM_pofd)
        NOAA_pofd_Full.append(NOAA_pofd)
        MO_pofd_Volcat_Full.append(MO_pofd_VOLCAT)
        BoM_pofd_Volcat_Full.append(BoM_pofd_VOLCAT)
        NOAA_pofd_Volcat_Full.append(NOAA_pofd_VOLCAT)
        
        MO_pofa_Full.append(MO_pofa)
        BoM_pofa_Full.append(BoM_pofa)
        NOAA_pofa_Full.append(NOAA_pofa)
        MO_pofa_Volcat_Full.append(MO_pofa_VOLCAT)
        BoM_pofa_Volcat_Full.append(BoM_pofa_VOLCAT)
        NOAA_pofa_Volcat_Full.append(NOAA_pofa_VOLCAT)
        
        n_MOsat_Full.append(n_sat_limit)
        n_Volcat_Full.append(n_volcat_limit)
        timestr_Full.append(timestep)
        plot_time_Full.append(plot_time)
       
        write_stats_file(hour_start_str, hour_end_str, n_sat, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, MO_FMS, BoM_FMS, NOAA_FMS, timestep, outdir, threshold, 'FMS')

        write_stats_file(hour_start_str, hour_end_str, n_sat, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, MO_precision, BoM_precision, NOAA_precision, timestep, outdir, threshold, 'precision')

        write_stats_file(hour_start_str, hour_end_str, n_sat, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, MO_pod, BoM_pod, NOAA_pod, timestep, outdir, threshold, 'pod')

        write_stats_file(hour_start_str, hour_end_str, n_sat, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, MO_pofd, BoM_pofd, NOAA_pofd, timestep, outdir, threshold, 'pofd')

        write_stats_file(hour_start_str, hour_end_str, n_sat, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, MO_pofa, BoM_pofa, NOAA_pofa, timestep, outdir, threshold, 'pofa')

        write_stats_file(hour_start_str, hour_end_str, n_volcat, n_volcat_limit, n_MO_volcat_limit, n_BoM_volcat_limit, n_NOAA_volcat_limit, MO_FMS_VOLCAT, BoM_FMS_VOLCAT, NOAA_FMS_VOLCAT, timestep, outdir, threshold_volcat, 'FMS', True)
        
        write_stats_file(hour_start_str, hour_end_str, n_volcat, n_volcat_limit, n_MO_volcat_limit, n_BoM_volcat_limit, n_NOAA_volcat_limit, MO_precision_VOLCAT, BoM_precision_VOLCAT, NOAA_precision_VOLCAT, timestep, outdir, threshold_volcat, 'precision', True)
        
        write_stats_file(hour_start_str, hour_end_str, n_volcat, n_volcat_limit, n_MO_volcat_limit, n_BoM_volcat_limit, n_NOAA_volcat_limit, MO_pod_VOLCAT, BoM_pod_VOLCAT, NOAA_pod_VOLCAT, timestep, outdir, threshold_volcat, 'pod', True)
        
        write_stats_file(hour_start_str, hour_end_str, n_volcat, n_volcat_limit, n_MO_volcat_limit, n_BoM_volcat_limit, n_NOAA_volcat_limit, MO_pofd_VOLCAT, BoM_pofd_VOLCAT, NOAA_pofd_VOLCAT, timestep, outdir, threshold_volcat, 'pofd', True)

        write_stats_file(hour_start_str, hour_end_str, n_volcat, n_volcat_limit, n_MO_volcat_limit, n_BoM_volcat_limit, n_NOAA_volcat_limit, MO_pofa_VOLCAT, BoM_pofa_VOLCAT, NOAA_pofa_VOLCAT, timestep, outdir, threshold_volcat, 'pofa', True)

        plot_stat(MO_FMS, BoM_FMS, NOAA_FMS, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, plot_time, outdir, timestep, 'MO', threshold, 'FMS')

        plot_stat(MO_precision, BoM_precision, NOAA_precision, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, plot_time, outdir, timestep, 'MO', threshold, 'precision')

        plot_stat(MO_pod, BoM_pod, NOAA_pod, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, plot_time, outdir, timestep, 'MO', threshold, 'pod')

        plot_stat(MO_pofd, BoM_pofd, NOAA_pofd, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, plot_time, outdir, timestep, 'MO', threshold, 'pofd')

        plot_stat(MO_pofa, BoM_pofa, NOAA_pofa, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, plot_time, outdir, timestep, 'MO', threshold, 'pofa')

        plot_stat(MO_FMS_VOLCAT, BoM_FMS_VOLCAT, NOAA_FMS_VOLCAT, n_volcat_limit, n_MO_volcat_limit, n_BoM_volcat_limit, n_NOAA_volcat_limit, plot_time, outdir, timestep, 'Volcat', threshold_volcat, 'FMS')

        plot_stat(MO_precision_VOLCAT, BoM_precision_VOLCAT, NOAA_precision_VOLCAT, n_volcat_limit, n_MO_volcat_limit, n_BoM_volcat_limit, n_NOAA_volcat_limit, plot_time, outdir, timestep, 'Volcat', threshold_volcat, 'precision')

        plot_stat(MO_pod_VOLCAT, BoM_pod_VOLCAT, NOAA_pod_VOLCAT, n_volcat_limit, n_MO_volcat_limit, n_BoM_volcat_limit, n_NOAA_volcat_limit, plot_time, outdir, timestep, 'Volcat', threshold_volcat, 'pod')

        plot_stat(MO_pofd_VOLCAT, BoM_pofd_VOLCAT, NOAA_pofd_VOLCAT, n_volcat_limit, n_MO_volcat_limit, n_BoM_volcat_limit, n_NOAA_volcat_limit, plot_time, outdir, timestep, 'Volcat', threshold_volcat, 'pofd')

        plot_stat(MO_pofa_VOLCAT, BoM_pofa_VOLCAT, NOAA_pofa_VOLCAT, n_volcat_limit, n_MO_volcat_limit, n_BoM_volcat_limit, n_NOAA_volcat_limit, plot_time, outdir, timestep, 'Volcat', threshold_volcat, 'pofa')

    plot_stat_all(MO_FMS_Full, BoM_FMS_Full, NOAA_FMS_Full, n_MOsat_Full, plot_time_Full, outdir, timestr_Full, 'MO', threshold, 'FMS')

    plot_stat_all(MO_precision_Full, BoM_precision_Full, NOAA_precision_Full, n_MOsat_Full, plot_time_Full, outdir, timestr_Full, 'MO', threshold, 'precision')

    plot_stat_all(MO_pod_Full, BoM_pod_Full, NOAA_pod_Full, n_MOsat_Full, plot_time_Full, outdir, timestr_Full, 'MO', threshold, 'pod')

    plot_stat_all(MO_pofd_Full, BoM_pofd_Full, NOAA_pofd_Full, n_MOsat_Full, plot_time_Full, outdir, timestr_Full, 'MO', threshold, 'pofd')

    plot_stat_all(MO_pofa_Full, BoM_pofa_Full, NOAA_pofa_Full, n_MOsat_Full, plot_time_Full, outdir, timestr_Full, 'MO', threshold, 'pofa')

    plot_stat_all(MO_FMS_Volcat_Full, BoM_FMS_Volcat_Full, NOAA_FMS_Volcat_Full, n_Volcat_Full, plot_time_Full, outdir, timestr_Full, 'Volcat', threshold_volcat, 'FMS')

    plot_stat_all(MO_precision_Volcat_Full, BoM_precision_Volcat_Full, NOAA_precision_Volcat_Full, n_Volcat_Full, plot_time_Full, outdir, timestr_Full, 'Volcat', threshold_volcat, 'precision')

    plot_stat_all(MO_pod_Volcat_Full, BoM_pod_Volcat_Full, NOAA_pod_Volcat_Full, n_Volcat_Full, plot_time_Full, outdir, timestr_Full, 'Volcat', threshold_volcat, 'pod')

    plot_stat_all(MO_pofd_Volcat_Full, BoM_pofd_Volcat_Full, NOAA_pofd_Volcat_Full, n_Volcat_Full, plot_time_Full, outdir, timestr_Full, 'Volcat', threshold_volcat, 'pofd')

    plot_stat_all(MO_pofa_Volcat_Full, BoM_pofa_Volcat_Full, NOAA_pofa_Volcat_Full, n_Volcat_Full, plot_time_Full, outdir, timestr_Full, 'Volcat', threshold_volcat, 'pofa')

            
def write_data_file(lonarr, latarr, satarr, Marr, Barr, Narr, timestep, timestart, outdir, volcat=False):

    if volcat:
        acl_file= outdir+"Fields_Volcat_"+timestep+"_VT"+timestart+".txt"
    else:
        acl_file = outdir+"Fields_MO_"+timestep+"_VT"+timestart+".txt"
        
    with open(acl_file, 'w') as f:

        if volcat:
            f.write(f'  latitude, longitude,      VOLCAT,          MO,         BoM,        NOAA\n')
        else:
            f.write(f'  latitude, longitude,MO satellite,          MO,         BoM,        NOAA\n')
        
        for ii in range(len(latarr)):
            f.write(f'{latarr[ii]:>10}, {lonarr[ii]:>10}, {satarr[ii]:>12.4e}, {Marr[ii]:>12.4e}, {Barr[ii]:>12.4e}, {Narr[ii]:>12.4e}\n')
        


def write_stats_file(hour_start, hour_end, n_sat, n_sat_limit, n_MO_limit, n_BoM_limit, n_NOAA_limit, MO_stat, BoM_stat, NOAA_stat, timestep, outdir, threshold, stat, volcat=False):

# Writes out ascii files of calculated statistics. 
# A separate file for each issue time is generated containing statsitcs for all 3 models and at different forecast times

    if volcat:
        stats_file = outdir+stat+"_Volcat_"+timestep+".txt"
    else:
        stats_file = outdir+stat+"_MO_"+timestep+".txt"
        
    with open(stats_file, 'w') as f:

        f.write(f'Statistic: {stat}\n')
        f.write(f'Issue time: {timestep[8:10]}:{timestep[10:12]} UTC {timestep[6:8]}/{timestep[4:6]}/{timestep[:4]}\n')
        f.write(f'Threshold = {str(threshold)}\n')
        f.write(f'          Hour start,             Hour end, # sat points,  # sat>thresh,   # MO>thresh,  # BoM>thresh, # NOAA>thresh,      MO,     BoM,    NOAA\n')
        
        for i in range(len(hour_start)):
            f.write(f'{hour_start[i]}, {hour_end[i]}, {n_sat[i]:>12}, {n_sat_limit[i]:>13}, {n_MO_limit[i]:>13}, {n_BoM_limit[i]:>13}, {n_NOAA_limit[i]:>13}, {MO_stat[i]:>7.2f}, {BoM_stat[i]:>7.2f}, {NOAA_stat[i]:>7.2f}\n')
        

def subselect_MO(MTotCol, BTotCol, NTotCol, MO_sat_ash, MO_sat_clear, t0):

    # Extracting lat-lon points and data for overlapping points
#    print('Extracting data for shared lat-lon points')

# Checking time of MO and BoM data are aligned
#    print(f'MO data time: {MTotCol.coord('time').units.num2date(MTotCol.coord('time').points)}')
    timediff = int(1+3*BTotCol.coord('time').points[0])
#    print(timediff)
    BoM_MO_time_check = MTotCol.coord('time').units.num2date(MTotCol.coord('time').points) - t0 - datetime.timedelta(hours=timediff)
    if BoM_MO_time_check[0].total_seconds() != 0.0:
        sys.exit("MO data and BoM data is not for the same time")
#    print(type(BoM_MO_time_check[0]))
#    print(f'BoM data time: {BTotCol.coord('time').points[0]}')
    
#    print(NTotCol.coord('latitude').points)
#    print('NOAA longitudes')
#    print(NTotCol.coord('longitude').points[0,:])
    lonarr=NTotCol.coord('longitude').points[0,:]
#    print(type(lonarr))
#    print('lonarr min: ',lonarr.min())
#    print('BoM longitudes')
#    print(BTotCol.coord('longitude').points)
#    print('MO longitudes')
#    print(MTotCol.coord('longitude').points)
    
#    print('n: ')
#    print(n for n in NTotCol.coord('longitude').points[0,:] if n>0.0)
#    print(numpy.where(lonarr==[max(lonarr[lonarr<0])]))
#    print(max(lonarr[lonarr<0]))
#    print(min(lonarr[lonarr>0]))
#    print(MO_sat_ash)
#    print(MO_sat_clear)
    latmin = NTotCol.coord('latitude').points.min()
    latmax = NTotCol.coord('latitude').points.max()
    lonmin_pos = min(lonarr[lonarr>0])
    lonmax_neg = max(lonarr[lonarr<0])
    
#    print('lonmin_pos: ',lonmin_pos,', lonmax_neg: ',lonmax_neg,', latmin: ',latmin,', latmax: ',latmin) 
    
# Extracting MO satellite, MO model data and BoM model data to align with NOAA model data

    M_sel = MTotCol.extract(iris.Constraint(coord_values={'latitude':lambda cell: latmin <= cell <= latmax, 'longitude':lambda cell: lonmin_pos <= cell or lonmax_neg >= cell} ))
    B_sel = BTotCol.extract(iris.Constraint(coord_values={'latitude':lambda cell: latmin <= cell <= latmax, 'longitude':lambda cell: lonmin_pos <= cell or lonmax_neg >= cell} ))
    ash_sel = MO_sat_ash.extract(iris.Constraint(coord_values={'latitude':lambda cell: latmin <= cell <= latmax, 'longitude':lambda cell: lonmin_pos <= cell or lonmax_neg >= cell} ))
    clear_sel = MO_sat_clear.extract(iris.Constraint(coord_values={'latitude':lambda cell: latmin <= cell <= latmax, 'longitude':lambda cell: lonmin_pos <= cell or lonmax_neg >= cell} ))
    
#    print(NTotCol)
#    print(M_sel)
#    print(B_sel)
#    print(ash_sel)
#    print(clear_sel)
    

#    print(M_sel.coord('longitude').points)
#    print(M_sel.coord('latitude').points)
#    print(NTotCol.coord('longitude').points[0,:])
#    print(NTotCol.coord('latitude').points[:,0])

    latarr=[]
    lonarr=[]
    satarr=[]
    Narr=[]
    Barr=[]
    Marr=[]
    

    for i in range(len(M_sel.data[:,0])):
        for j in range(len(M_sel.data[0,:])):
            # Remove missing satellite values from comparison (MO_sat_ash and MO_sat_clear are both zero)
            if ash_sel.data[i,j] > 0.0 or clear_sel.data[i,j] > 0.0:
                NOAA_j = -999
                N_lon = NTotCol.coord('longitude').points[0,:]
#                print(type(N_lon))
#                print('N_lon: ',N_lon)
                for k in range(len(NTotCol.coord('longitude').points[0,:])):
                    if NTotCol.coord('longitude').points[0,k] == M_sel.coord('longitude').points[j]:
                        NOAA_j = k
                        break
                if NOAA_j < 0:
                    print(f'NOAA point not found for latitude {M_sel.coord('longitude').points[j]}')
#                print('i: ',i,' j: ',j,', NOAA_j: ',NOAA_j,', len: ',len(M_sel.data[0,:]))
#                print('M longitude: ',M_sel.coord('longitude').points[j],', N longitude: ',NTotCol.coord('longitude').points[0,NOAA_j])
            

# Check grids are aligned
                if ash_sel.coord('latitude').points[i] != clear_sel.coord('latitude').points[i] or ash_sel.coord('latitude').points[i] != M_sel.coord('latitude').points[i] or ash_sel.coord('latitude').points[i] != B_sel.coord('latitude').points[i] or ash_sel.coord('latitude').points[i] != NTotCol.coord('latitude').points[i,0]:
#                    print(f'MO ash satellite data, latitude: {ash_sel.coord('latitude').points[i]} and {clear_sel.coord('latitude').points[i]}')
#                    print(f'MO model data, latitude: {M_sel.coord('latitude').points[i]} and BoM model data, latitude: {B_sel.coord('latitude').points[i]}')
#                    print(f'NOAA model data, latitude: {NTotCol.coord('latitude').points[i,0]}')
                    sys.exit("MO satellite data, MO model data, BoM model data and NOAA model data are not on the same latitude grid")

                if ash_sel.coord('longitude').points[j] != clear_sel.coord('longitude').points[j] or ash_sel.coord('longitude').points[j] != M_sel.coord('longitude').points[j] or ash_sel.coord('longitude').points[j] != B_sel.coord('longitude').points[j] or ash_sel.coord('longitude').points[j] != NTotCol.coord('longitude').points[0,NOAA_j]:
#                    print(f'MO ash satellite data, longitude: {ash_sel.coord('longitude').points[j]} and {clear_sel.coord('longitude').points[j]}')
#                    print(f'MO model data, longitude: {M_sel.coord('longitude').points[j]} and BoM model data, longitude: {B_sel.coord('longitude').points[j]}')
#                    print(f'NOAA model data, longitude: {NTotCol.coord('longitude').points[0,NOAA_j]}')
                    sys.exit("MO satellite data, MO model data, BoM model data and NOAA model data are not on the same longitudegrid")
 
                lonarr.append(ash_sel.coord('longitude').points[j])
                latarr.append(ash_sel.coord('latitude').points[i])
                
                if ash_sel.data[i,j] > 0.0:
                    satarr.append(ash_sel.data[i,j])
                else:
                    satarr.append(0.0)
                   
                Marr.append(M_sel.data[i,j])
                Barr.append(B_sel.data[i,j])
#                print(f'Narr value: {NTotCol.data[:,:,i,NOAA_j][0][0]}')
                Narr.append(NTotCol.data[:,:,i,NOAA_j][0][0])

    return(lonarr, latarr, satarr, Marr, Barr, Narr)


def calc_stats(satarr, Modelarr, threshold):
    
    # Calculates statistics
#    print('Calculating statistics')
    
# Contingency table
# a: model true, obs true
# b: model true, obs false
# c: model false, obs true
# d: model false, obs false
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    
    n_sat_points = 0
    n_model_threshold = 0
    n_sat_threshold = 0
        

    for ii in range(len(satarr)):    
 
 # Calculate FMS
        n_sat_points = n_sat_points + 1

        if Modelarr[ii] >= threshold:
            n_model_threshold = n_model_threshold + 1
            if satarr[ii] >= threshold:
                n_sat_threshold = n_sat_threshold + 1
                a = a + 1.0
            else:
                b = b + 1.0
        else:
            if satarr[ii] >= threshold:
                n_sat_threshold = n_sat_threshold + 1
                c = c + 1.0
            else:
                d = d + 1.0

    
#    print(f'Number of MO satellite points: {n_sat_points}')

    if a + b + c > 0:
        fms = 100.0 * (a / (a + b + c))
    else:
        fms = numpy.nan
#    print(f'fms: {fms}')
#    print(f'Number in intersection: {a}')
#    print(f'Number in union: {a + b + c}')
    
  # Calculate Precision
    if a + b > 0:
        precision = 100.0 * (a / (a + b))
    else:
        precision = numpy.nan
#    print(f'precision: {precision}')

  # Calculate Probability of detection (POD)
    if a + c > 0:
        pod = 100.0 * (a / (a + c))
    else:
        pod = numpy.nan
#    print(f'pod: {pod}')

  # Calculate Probability of false detection (POFD)
    if b + d > 0:
        pofd = 100.0 * (b / (b + d))
    else:
        pofd = numpy.nan
#    print(f'pofd: {pofd}')

  # Calculate Probability of false alarm (POFA)
    if a + b > 0:
        pofa = 100.0 * (b / (a + b))
    else:
        pofa = numpy.nan
#    print(f'pofa: {pofa}')

#    print('')
  

    return(fms, precision, pod, pofd, pofa, n_sat_points, n_sat_threshold, n_model_threshold)
 
def subselect_Volcat(MTotCol, BTotCol, NTotCol, Volcat, t0):
    
    # Extracting lat-lon points and data for overlapping points
#    print('Extracting data for shared lat-lon points with Volcat data')

# Checking time of MO and BoM data are aligned
#    print(f'MO data time: {MTotCol.coord('time').units.num2date(MTotCol.coord('time').points)}')
    timediff = int(1+3*BTotCol.coord('time').points[0])
#    print(timediff)
    BoM_MO_time_check = MTotCol.coord('time').units.num2date(MTotCol.coord('time').points) - t0 - datetime.timedelta(hours=timediff)
    if BoM_MO_time_check[0].total_seconds() != 0.0:
        sys.exit("MO data and BoM data is not for the same time")
#    print(type(BoM_MO_time_check[0]))
#    print(f'BoM data time: {BTotCol.coord('time').points[0]}')
    
#    print(NTotCol)
    
# Latitude-longitude information
#    print('NOAA longitudes')
#    print(NTotCol.coord('longitude').points[0,:])
    NOAA_lonarr=NTotCol.coord('longitude').points[0,:]
#    print('BoM longitudes')
#    print(BTotCol.coord('longitude').points)
    BoM_lonarr=BTotCol.coord('longitude').points
#    print('MO longitudes')
#    print(MTotCol.coord('longitude').points)
    MO_lonarr=MTotCol.coord('longitude').points
#    print('Volcat longitudes')
#    print(Volcat.coords['longitude'][0,:].values)
    Volcat_lonarr=Volcat.coords['longitude'][0,:].values


#    print('NOAA latitudes')
#    print(NTotCol.coord('latitude').points[:,0])
    NOAA_latarr=NTotCol.coord('latitude').points[:,0]
#    print('BoM latitudes')
#    print(BTotCol.coord('latitude').points)
    BoM_latarr=BTotCol.coord('latitude').points
#    print('MO latitudes')
#    print(MTotCol.coord('latitude').points)
    MO_latarr=MTotCol.coord('latitude').points
#    print('Volcat latitudes')
#    print(Volcat.coords['latitude'][:,0].values)
    Volcat_latarr=Volcat.coords['latitude'][:,0].values
#    print(len(Volcat.coords['latitude'][:,0].values))
#    print('Volcat latitude min: ',Volcat.coords['latitude'][:,0].min())
#    print('Volcat latitude max: ',Volcat.coords['latitude'][:,0].max())

    latarr=[]
    lonarr=[]
    satarr=[]
    Narr=[]
    Barr=[]
    Marr=[]

#    print('# Volcat latitudes: ',len(Volcat_latarr))
#    print('# Volcat longitudes: ',len(Volcat_lonarr))
#    print(f'Shape of MTotCol data: {numpy.shape(MTotCol.data)}')
    for ii in range(len(Volcat_latarr)):
        MO_ii=numpy.where(MO_latarr == Volcat_latarr[ii])
        BoM_ii=numpy.where(BoM_latarr == Volcat_latarr[ii])
        NOAA_ii=numpy.where(NOAA_latarr == Volcat_latarr[ii])
        if len(MO_ii[0]) == 1 and len(BoM_ii[0]) == 1 and len(NOAA_ii[0]) == 1:
            M_ii=MO_ii[0][0]
            B_ii=BoM_ii[0][0]
            N_ii=NOAA_ii[0][0]
#            print(f'length of MO_ii: {len(MO_ii[0])}')
#            print(f'MO_ii: {MO_ii[0]}')
#            print(f'1st element of MO_ii: {MO_ii[0][0]}')
#            print(f'M_ii: {M_ii}')
#            print(f'Volcat latitude: {Volcat_latarr[ii]}')
#            print(f'MO latitude: {MO_latarr[M_ii]}')

            if Volcat_latarr[ii] != MO_latarr[M_ii] or Volcat_latarr[ii] != BoM_latarr[B_ii] or Volcat_latarr[ii] != NOAA_latarr[N_ii]:
#                print(f'latitudes: VOLCAT {Volcat_latarr[ii]}, MO {MO_latarr[M_ii]}, BoM {BoM_latarr[B_ii]}, NOAA {NOAA_latarr[N_ii]}')
                sys.exit("Latitudes do not agree")

            for jj in range(len(Volcat_lonarr)):
                MO_jj=numpy.where(MO_lonarr == Volcat_lonarr[jj])
                BoM_jj=numpy.where(BoM_lonarr == Volcat_lonarr[jj])
                NOAA_jj=numpy.where(NOAA_lonarr == Volcat_lonarr[jj])
                if len(MO_jj[0]) == 1 and len(BoM_jj[0]) == 1 and len(NOAA_jj[0]) == 1:
                    M_jj=MO_jj[0][0]
                    B_jj=BoM_jj[0][0]
                    N_jj=NOAA_jj[0][0]
                    if Volcat_lonarr[jj] != MO_lonarr[M_jj] or Volcat_lonarr[jj] != BoM_lonarr[B_jj] or Volcat_lonarr[jj] != NOAA_lonarr[N_jj]:
                        print(f'longitudes: VOLCAT {Volcat_lonarr[jj]}, MO {MO_lonarr[M_jj]}, BoM {BoM_lonarr[B_jj]}, NOAA {NOAA_lonarr[N_jj]}')
                        sys.exit("Longitudes do not agree")
 #                   print(f'M_jj: {M_jj}')
 #                   print(f'Volcat longitude: {Volcat_lonarr[jj]}')
 #                   print(f'MO longitude: {MO_lonarr[M_jj]}')
            
 #                   print(f'Volcat data value: {Volcat.values[ii,jj]}')
 #                   print(f'MO data value: {MTotCol.data[M_ii,M_jj]}')
                    if MTotCol.data[M_ii,M_jj] > 0.0 or BTotCol.data[B_ii,B_jj] > 0.0 or NTotCol.data[:,:,N_ii,N_jj][0][0] > 0.0 or Volcat.values[ii,jj] > 0.0:
                        lonarr.append(Volcat_lonarr[jj])
                        latarr.append(Volcat_latarr[ii])
                        Marr.append(MTotCol.data[M_ii,M_jj])
                        Barr.append(BTotCol.data[B_ii,B_jj])
                        Narr.append(NTotCol.data[:,:,N_ii,N_jj][0][0])
                        satarr.append(Volcat.values[ii,jj])
           
#    print('lonarr: ',lonarr)
#    print('latarr: ',latarr)
#    print('satarr: ',satarr)
#    print('Marr: ',Marr)
#    print('Barr: ',Barr)
#    print('Narr: ',Narr)

    return(lonarr, latarr, satarr, Marr, Barr, Narr)
 

        
def read_MO_data(M_dir, timestep):
    
#    print('Reading MO results')
    Mcube = iris.load_cube(M_dir+"NAME_InTEM_forecast_"+timestep+".nc")
        
#    print(Mcube)

# Calculate column loads
    MTotCol = Mcube.collapsed('flight_level',iris.analysis.SUM)
#    print(MTotCol)
#    print(MTotCol.dim_coords)
#    print(MTotCol.data.max())

# Multiply by layer depth and convert from FL to m and from mg to g
    MTotCol.data = MTotCol.data * 5.0 * 0.3048
    MTotCol.units = "g / m^2"
#    print(MTotCol)
#    print(MTotCol.coord('time'))
    Mtime = MTotCol.coord('time')
#    print(repr(Mtime.units))
#    print(Mtime.points)
#    print(Mtime.bounds)
    
#    print(MTotCol.coord('latitude'))
#    print(MTotCol.coord('longitude'))
#    print(Mtime.units.num2date(Mtime.points))

#    print(MTotCol.data.max())
        
#    print(MTotCol.units)

    return MTotCol

def read_BoM_data(B_dir, timestep):
  
#    print('Reading BoM results')
    # BoM has to be loaded as an xarray first
    Bxarray = xr.open_dataset(B_dir+"forecast_"+timestep[:8]+"T"+timestep[8:]+"Z_gl.nc")
    Bxarray['concentration'].coords['levels'].attrs['standard_name'] = ''
    Bxarray['concentration'].attrs['standard_name'] = ''
    Bcube = Bxarray['concentration'].to_iris()
    Bcube.coord('flight level of model layer mid-point').rename('flight_level')

# Calculate column loads
    BTotCol = Bcube.collapsed('flight_level',iris.analysis.SUM)
#    print(BTotCol)
#    print(BTotCol.dim_coords)
#    print(BTotCol.data.max())
#    print(BTotCol.units)
# Multiply by layer depth and convert from FL to m and from mg to g
    BTotCol.data = BTotCol.data * 5.0 * 0.3048
    BTotCol.units = "g / m^2"
#    print(BTotCol)
#    print(BTotCol.data.max())
        
#    print(BTotCol.units)
 
#    print(BTotCol)
#    print(BTotCol.coord('time'))
    Btime = BTotCol.coord('time')

#    print(repr(Btime.units))
#    print(Btime.points)
#    print(Btime.bounds)
 
#    print(BTotCol.coord('latitude'))
#    print(BTotCol.coord('longitude'))
       
    return BTotCol


def read_NOAA_data(N_dir, time):

#    print('Reading NOAA data')
    
#    print(N_dir)
    
#    print(timestep)
    
    first_issue_time = datetime.datetime(2021, 11, 3, 9, 0)
    
    timediff = time - first_issue_time
    hourdiff = divmod(timediff.total_seconds(), 3600)[0]
    
    if hourdiff == 0:
        modelrun = '1'
    else:
        modelrun = str(int((hourdiff -3)/6+2))
        
#    print('NOAA model run: ',modelrun)
     
    Ncube = iris.load_cube(N_dir+"HYSPLIT_inv_forecast_"+modelrun+".nc")

# Calculate column loads
    NTotCol = Ncube.collapsed('z',iris.analysis.SUM)
#    print(NTotCol)
#    print(NTotCol.units)
#    print(NTotCol.dim_coords)
#    print(NTotCol.data.max())
#    print(NTotCol.data.min())
#    print(NTotCol.coord('latitude'))
#    print(NTotCol.coord('longitude'))

    NTotCol.data = NTotCol.data * 5000.0 * 0.3048
    NTotCol.units = "g / m^2"
    
    
#    N_das= xr.open_dataset(N_dir+"HYSPLIT_inv_forecast_1.nc")
    
#    print(N_das)
    
    return(NTotCol)
    

def read_volcat_data(VOLCAT_sat_dir, hour_start, hour_end):

#    print('Reading VOLCAT observations')
    xr.open_dataset

# get all VOLCAT Files between two dates
    drange = None
    flist = glob.glob(VOLCAT_sat_dir + '*nc')
    das = volcat.get_volcat_list(VOLCAT_sat_dir,flist=None,verbose=True,daterange=[hour_start,hour_end],include_last=False)
    
 # get dataframe with info from all the files.
    df = volcat.get_volcat_name_df(VOLCAT_sat_dir)
    
# add the files into one data-array with a time axis
    dset = volcat.combine_regridded(das)


# perform the time averaging (only on the mass loading field)
    dset_averaged = dset.ash_mass_loading.mean(dim='time')

#    print(dset_averaged)   
    
#    print('dset')
#    print(type(dset_averaged))
#    print(dset_averaged)
    
    dset_averaged_withoutnan = dset_averaged.fillna(0.0)
#    nan_array = numpy.isnan(dset_averaged.data)
    
#    print(nan_array)
    
#    print(dset_averaged.data.size)
#    print(numpy.nanmax(dset_averaged.data))
#    print(numpy.nanmin(dset_averaged.data))
#    print(dset_averaged.coords)
#    print(dset_averaged.coords['latitude'])
#    print(dset_averaged.coords['latitude'][0])
#    print(dset_averaged.coords['latitude'].data.min())
#    print(dset_averaged.coords['latitude'].data.max())
#    print(dset_averaged.coords['longitude'])
#    print(dset_averaged.coords['longitude'].data.min())
#    print(dset_averaged.coords['longitude'].data.max())
#    print(dset_averaged.sizes['x'])
    
    return dset_averaged_withoutnan


def read_MO_sat_data(MO_sat_dir, hour_end):
    
    print('Reading MO satellite observations')
    timevar = hour_end.strftime('%Y%m%d%H%M')
    
    print(timevar)
    
    filename_sat = MO_sat_dir+'HIM8_'+timevar+'_va_out.csv'
    filename_clear = MO_sat_dir+'HIM8_'+timevar+'_cloudfree_out.csv'
    
#    print(filename_sat)
#    print(filename_clear)
   
    ashcube = obs_loader_ALL.obs_to_cube(filename_sat)      
#    print(ashcube)
#    print('MO sat ash latitude')
#    print(ashcube.coord('latitude').points[0])
#    print('MO sat ash longitude')
#    print(ashcube.coord('longitude'))
#    print(numpy.max(ashcube.data))
#    print(numpy.min(ashcube.data))
    
    clearskycube = obs_loader_ALL.obs_to_cube(filename_clear,clear_sky=True)     		
#    print(clearskycube)
#    print(clearskycube.coord('latitude'))
#    print(clearskycube.coord('longitude'))
#    print(numpy.max(clearskycube.data))
#    print(numpy.min(clearskycube.data))
    
    return ashcube, clearskycube

def plot_stat(MO_stat, BoM_stat, NOAA_stat, n_sat, n_MO, n_BoM, n_NOAA, hour_start, outdir, timestep, obs, threshold, stat):
    
# Generates statistics plots for each issue time
    myFmt = mdates.DateFormatter('%m-%d %H:%M') 

    # Generate stats plot against satellite data
    fig, axs = plt.subplots(1)
    axs2 = axs.twinx()
    xloc = mdates.HourLocator(interval = 3)
    axs.xaxis.set_major_locator(xloc)
    axs.xaxis.set_major_formatter(myFmt)
    axs.plot(hour_start,MO_stat,color='green',marker = 's', label='MO')
    axs.plot(hour_start,BoM_stat,color='blue',marker = 'o', label='BoM')
    axs.plot(hour_start,NOAA_stat,color='red',marker = '^', label='NOAA')
    axs2.plot(hour_start, n_sat, color='grey', linewidth=0.5, label='# sat > threshold')
    axs2.plot(hour_start, n_MO, color='green', linewidth=0.5, label='# MO > threshold')
    axs2.plot(hour_start, n_BoM, color='blue', linewidth=0.5, label='# BoM > threshold')
    axs2.plot(hour_start, n_NOAA, color='red', linewidth=0.5, label='# NOAA > threshold')

    axs.set_ylabel(stat)
    axs2.set_ylabel('# > threshold')
 
    if stat == 'precision':
        axs.legend(loc='lower right')
        axs2.legend(loc='lower center')
    
    else:
        axs.legend(loc='upper right')
        axs2.legend(loc='upper center')

    title_str = 'Issue time: '+timestep[8:10]+':'+timestep[10:]+'Z '+timestep[6:8]+'/'+timestep[4:6]+'/'+timestep[:4]+'\n '+obs+' satellite data with threshold '+str(threshold)

    axs.set_xlabel('Time')
    axs.set_title(title_str)
    axs.tick_params(axis='x', labelrotation=90) 

    figname = stat+'_'+obs+'_'+timestep+'.png'
    axs.set_ylim(0.0,100.0)
    if obs == 'Volcat':
        if threshold >= 0.5:
            axs2.set_ylim(0,550)        
        else:
            axs2.set_ylim(0,950)
    elif obs == 'MO':
        axs2.set_ylim(0,100)
    plt.tight_layout()
    fig.savefig(outdir+ figname)
    plt.close('all')

def plot_stat_all(MO_stat, BoM_stat, NOAA_stat, n_sat, hour_start, outdir, timestep, obs, threshold, stat):
    
# Generates statistics plots for all issue times
    myFmt = mdates.DateFormatter('%m-%d %H:%M') 
    
    colours_MO = ['palegreen','greenyellow','lawngreen','limegreen','mediumseagreen','seagreen','green','darkgreen']
    colours_BoM = ['lightblue','deepskyblue','steelblue','cornflowerblue','royalblue','mediumblue','darkblue','midnightblue']
    colours_NOAA = ['violet','darkviolet','purple','maroon','firebrick','red','salmon','lightpink']

    # Generate stats plot against satellite data for all issue times
    fig, axs = plt.subplots(1)
    axs2 = axs.twinx()
    axs.xaxis.set_major_formatter(myFmt)
    for i in range(len(timestep)):
        axs.plot(hour_start[i],MO_stat[i],color=colours_MO[i],marker = 's')
        axs.plot(hour_start[i],BoM_stat[i],color=colours_BoM[i],marker = 'o')
        axs.plot(hour_start[i],NOAA_stat[i],color=colours_NOAA[i],marker = '^')
        axs2.plot(hour_start[i], n_sat[i], color='black')

    axs.set_ylabel(stat)
    axs2.set_ylabel('# satellite points')
 
#    if stat == 'pofd' or stat == 'precision':
#        axs2.legend(loc='lower right')
#    elif:
#        axs2.legend(loc='upper right')

    title_str = obs+' satellite data with threshold '+str(threshold)

    axs.set_xlabel('Time')
    axs.set_title(title_str)
    axs.tick_params(axis='x', labelrotation=90) 

    figname = stat+'_'+obs+'.png'
    axs.set_ylim(0.0,100.0)
    plt.tight_layout()
    fig.savefig(outdir+ figname)
    plt.close('all')

if __name__ == "__main__":

    # set file locations and dates
    MO_dir = "/data/users/apdg/projects/INTEM/VAAC_Intercomparison/Karymsky_point25/MetOffice_results_Aug2025/"
    BoM_dir = "/data/users/apdg/projects/INTEM/VAAC_Intercomparison/Karymsky_BoM/NewData_Nov2024/"
    NOAA_dir = "/data/users/apdg/projects/INTEM/VAAC_Intercomparison/Karymsky_NOAA/"
    MO_sat_dir = "/data/users/apdg/projects/INTEM/VAAC_Intercomparison/Karymsky_point25/SatData/"
    VOLCAT_sat_dir = "/data/users/apdg/projects/INTEM/VAAC_Intercomparison/Karymsky_point25/Alice/Volcat_Data/"
    outdir = "stats/"
    
    threshold = 0.5
    threshold_volcat = 0.2

    forecasts = ["202111030900","202111031200","202111031800","202111040000","202111040600","202111041200","202111041800","202111050000"]
#    forecasts = ["202111050000","202111050600","202111051200","202111051800"]
#    forecasts = ["202111031200"]


    Model_Obs_compare(MO_dir, BoM_dir, NOAA_dir, MO_sat_dir, VOLCAT_sat_dir, forecasts, outdir, threshold, threshold_volcat)

