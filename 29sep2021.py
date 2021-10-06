#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from brpylib import NevFile, brpylib_ver, NsxFile

brpylib_ver_req = "1.3.1"
if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
    raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")
nev_file = NevFile('datafile001.nev')
nev_data = nev_file.getdata()
ns3_file = NsxFile('datafile001.ns3')
ns3_data = ns3_file.getdata()
ns4_file = NsxFile('datafile001.ns4')
ns4_data = ns4_file.getdata()
nev_file.close()
ns3_file.close()
ns4_file.close()


# In[10]:


ns3_data.keys()


# In[11]:


plt.plot(ns3_data['data'][0])


# In[12]:


plt.plot(ns3_data['data'][1])


# In[ ]:





# In[13]:


plt.plot(ns3_data['data'][2])


# In[14]:


plt.plot(ns3_data['data'][3])


# In[15]:


plt.plot(ns3_data['data'][4])


# In[60]:


import matplotlib.pyplot as plt
import numpy as np

plt.figure(0, figsize=(20, 10))

plt.title('Simultaneous representation of ns3 data')

# How to make loop here, so that I won't need to write this same code for 40 times?!

plt.subplot(20,1,1)
plt.plot(ns3_data['data'][0])
plt.subplot(20,1,2)
plt.plot(ns3_data['data'][1])
plt.subplot(20,1,3)
plt.plot(ns3_data['data'][2])
plt.subplot(20,1,4)
plt.plot(ns3_data['data'][3])
plt.subplot(20,1,5)
plt.plot(ns3_data['data'][4])
plt.subplot(20,1,6)
plt.plot(ns3_data['data'][5])
plt.subplot(20,1,7)
plt.plot(ns3_data['data'][6])
plt.subplot(20,1,8)
plt.plot(ns3_data['data'][7])
plt.subplot(20,1,9)
plt.plot(ns3_data['data'][8])
plt.subplot(20,1,10)
plt.plot(ns3_data['data'][9])
plt.subplot(20,1,11)
plt.plot(ns3_data['data'][10])
plt.subplot(20,1,12)
plt.plot(ns3_data['data'][11])
plt.subplot(20,1,13)
plt.plot(ns3_data['data'][12])
plt.subplot(20,1,14)
plt.plot(ns3_data['data'][13])
plt.subplot(20,1,15)
plt.plot(ns3_data['data'][14])
plt.subplot(20,1,16)
plt.plot(ns3_data['data'][15])
plt.subplot(20,1,17)
plt.plot(ns3_data['data'][16])
plt.subplot(20,1,18)
plt.plot(ns3_data['data'][17])
plt.subplot(20,1,19)
plt.plot(ns3_data['data'][18])
plt.subplot(20,1,20)
plt.plot(ns3_data['data'][19])





plt.show()


# In[ ]:




