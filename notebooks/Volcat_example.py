#!/usr/bin/env python
# coding: utf-8

# In[4]:




# In[5]:


import glob
import datetime
import os
import sys


# In[13]:


import volcat
import xarray as xr


# In[14]:


# replace with directory of volcat files.
tdir = '/hysplit3/alicec/projects/karymsky/volcat2/pc_corrected/'


# In[15]:


os.path.exists(tdir)


# In[48]:


# get all VOLCAT Files between two dates
drange = None
d1 = datetime.datetime(2021,11,3,8)
d2 = datetime.datetime(2021,11,3,9)
flist = glob.glob(tdir + '*nc')
das = volcat.get_volcat_list(tdir,flist=None,verbose=True, daterange=[d1,d2])


# In[49]:


print(len(das))


# In[50]:


# just open one volcat file with xarray
dset = xr.open_dataset(flist[10],engine='netcdf4')


# In[51]:


dset.ash_mass_loading.isel(time=0).plot.pcolormesh(x='longitude',y='latitude')


# In[52]:


# get dataframe with info from all the files.
df = volcat.get_volcat_name_df(tdir)


# In[53]:


print(df)


# In[54]:


das


# In[55]:


# add the files into one data-array with a time axis
dset = volcat.combine_regridded(das)


# In[56]:

# perform the time averaging (only on the mass loading field)
dset_averaged = dset.ash_mass_loading.mean(dim='time')


# In[57]:

# plot
dset_averaged.plot.pcolormesh(x='longitude',y='latitude')


# In[ ]:

# mass loading

# array with total mass from each time period 
# THis is directly from the volcat files which come with total mass
total_mass=[]
for d in das:
    total_mass.append(float(d.ash_mass_loading_total_mass.values)) 


# another way which multiplies area by mass loading
# field in data array must be in g/m2.
# returns mass in Tg
volcat.check_total_mass(dset_averaged)




