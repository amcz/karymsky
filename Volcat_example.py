#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[5]:


import glob
import datetime
import os
import sys


# In[13]:


import volcat
import xarray as xr


# In[14]:


tdir = '/hysplit3/alicec/projects/karymsky/volcat2/pc_corrected/'


# In[15]:


os.path.exists(tdir)


# In[16]:


xr.open_dataset


# In[31]:


get_ipython().run_line_magic('autoreload', '')
# get all VOLCAT Files between two dates
drange = None
d1 = datetime.datetime(2021,11,3,7)
d2 = datetime.datetime(2021,11,3,8)
flist = glob.glob(tdir + '*nc')
das = volcat.get_volcat_list(tdir,flist=None,verbose=True, daterange=[d1,d2])


# In[32]:


print(len(das))


# In[33]:


# just open one volcat file with xarray
dset = xr.open_dataset(flist[10],engine='netcdf4')


# In[34]:


dset.ash_mass_loading.isel(time=0).plot.pcolormesh(x='longitude',y='latitude')


# In[35]:


# get dataframe with info from all the files.
df = volcat.get_volcat_name_df(tdir)


# In[27]:


print(df)


# In[ ]:




