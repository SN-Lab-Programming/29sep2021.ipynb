#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from brpylib import NevFile, brpylib_ver, NsxFile
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq


# In[3]:


brpylib_ver_req = "1.3.1"
if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
    raise Exception("requires brpylib " +
                        brpylib_ver_req +
                        " or higher, please use latest version"
                    )


# In[4]:


datafile_nsx_pre = "/media/kenichirotsutsui/HDCZ-UT/data/71/20181217/datafile001.ns4"
datafile_nev_pre = "/media/kenichirotsutsui/HDCZ-UT/data/71/20181217/datafile001.nev"
datafile_nsx_post = "/media/kenichirotsutsui/HDCZ-UT/data/71/20181217/datafile002.ns4"
datafile_nev_post = "/media/kenichirotsutsui/HDCZ-UT/data/71/20181217/datafile002.nev"

nsx_file_pre = NsxFile(datafile_nsx_pre)
nev_file_pre = NevFile(datafile_nev_pre)
nsx_file_post = NsxFile(datafile_nsx_post)
nev_file_post = NevFile(datafile_nev_post)

nsx_pre = nsx_file_pre.getdata()
nev_pre = nev_file_pre.getdata()
nsx_post = nsx_file_post.getdata()
nev_post = nev_file_post.getdata()


# In[5]:


nsx_file_pre.close()
nev_file_pre.close()
nsx_file_post.close()
nev_file_post.close()


# In[25]:


nev_pre['dig_events']


# In[7]:


nsx_post['elec_ids']


# In[7]:


plt.figure(figsize=(20,3))
for i in range(5):
    plt.subplot(6,1,i+1)
    plt.plot(nsx_pre['data'][i])
plt.subplot(6,1,6)
plt.plot(nsx_pre['data'][-1])


# In[ ]:


plt.figure(figsize=(20,3))
plt.plot(nsx_post['data'][-1])


# In[ ]:


plt.figure(figsize=(20,3))

plt.subplot(411)
plt.plot(nsx_post['data'][0][0:36000000])
plt.subplot(412)
plt.plot(nsx_post['data'][1][0:36000000])
plt.subplot(413)
plt.plot(nsx_post['data'][2][0:36000000])
plt.subplot(414)
plt.plot(nsx_post['data'][-1][0:36000000])


# In[239]:


event_tmp = nev_post['dig_events']['TimeStamps'][0]


# In[240]:


event = []
for i in range(0,len(event_tmp),2):
    event.append(int(event_tmp[i]/3))


# In[242]:


len(event)


# In[243]:


trials_post = []
for i in event:
    trials_post.append(nsx_post['data'][-1][i-300:i+500])


# In[256]:


len(trials_post)


# In[252]:


fig3 = plt.figure(figsize=(35,100))
for i in range(len(trials_post)):
    plt.subplot(20,6,i+1)
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / ms')
    plt.ylabel('Voltage / uV')
    plt.ylim((-1000, 2000))
    plt.plot(x_lim,trials_post[i])
fig3.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/postTrials.png')


# In[257]:


for i in range(len(trials_post)):
    fig4 = plt.figure(figsize=(20,15))
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / ms')
    plt.ylabel('Voltage / uV')
    plt.ylim((-1000, 2000))
    plt.plot(x_lim,trials_post[i])
    fig4.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/Trials_'+str(i+1)+'_post.png')


# In[255]:


ave_trials1 = np.zeros(shape=len(trials_post[0]))
for i in range(len(trials)):
    ave_trials1 += np.array(trials_post[i])
ave_trials1 /= len(trials_post)
im = plt.figure(figsize=(20,15))
plt.title('TrialAverage')
plt.xlabel('Time / ms')
plt.ylabel('Voltage / uV')
plt.ylim((-1000, 2000))
plt.plot(x_lim,ave_trials1)
im.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/Trials_Ave_post.png')


# In[8]:


event_tmp = nev_pre['dig_events']['TimeStamps'][0]


# In[20]:


step = 20
fs = nsx_pre['samp_per_s']


# In[36]:


event = []
for i in range(0,len(event_tmp),2):
    event.append(int(event_tmp[i]/3))


# In[37]:


event


# In[38]:


np.argmax(nsx_pre['data'][-1])


# In[56]:


trials = []
for i in event:
    trials.append(nsx_pre['data'][-1][i-300:i+500])


# In[57]:


trials


# In[75]:


fig = plt.figure(figsize=(35,50))
for i in range(len(trials)):
    plt.subplot(10,6,i+1)
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / ms')
    plt.ylabel('Voltage / uV')
    plt.plot(x_lim,trials[i])
fig.savefig('/home/kenichirotsutsui/Documents/Zhao/Trials.png')


# In[12]:


data = []
for i in event:
    if i>len(nsx_pre['data'][0]):
        break
    data.append(nsx_pre['data'][0][i-300:i+500])


# In[70]:


x_lim=np.arange(-30,50,0.1)


# In[65]:


plt.figure(figsize=(10,3))
for i in range(21):
    plt.subplot(7,3,i+1)
    plt.plot(x_lim,data[i])


# In[15]:


len(data)


# In[22]:


from scipy import signal
 


# In[23]:


nsx_pre['samp_per_s']


# In[24]:


sample_f = nsx_pre['samp_per_s']


# In[25]:


b, a = signal.butter(2, [3/sample_f,500/sample_f], 'bandpass')   #配置滤波器 8 表示滤波器的阶数


# In[26]:


filtedData = []
for i in range(len(data4)):
    filtedData.append(signal.filtfilt(b, a, data4[i]))


# In[28]:


for i in range(21):
    plt.subplot(7,3,i+1)
    plt.plot(x_lim,filtedData[i])
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.8,hspace=0.3)


# In[81]:


data_mep = []
for i in event:
    if i>len(nsx_pre['data'][-1]):
        break
    data_mep.append(nsx_pre['data'][-1][i-300:i+500])


# In[82]:


len(data_mep)


# In[119]:


plt.figure(figsize=(10,6))
for i in range(21):
    plt.subplot(7,3,i+1)
    plt.plot(x_lim,data_mep[i])
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=1,hspace=1)


# In[121]:


bb, aa = signal.butter(2, [3/sample_f,500/sample_f], 'bandpass')   #配置滤波器 8 表示滤波器的阶数


# In[122]:


filtedData_mep = []
for i in range(len(data_mep)):
    filtedData_mep.append(signal.filtfilt(bb, aa, data_mep[i]))


# In[123]:


plt.figure(figsize=(10,6))
for i in range(len(filtedData_mep)):
    plt.subplot(7,3,i+1)
    plt.plot(x_lim,filtedData_mep[i])
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.8,hspace=0.3)


# In[40]:


plt.plot(nsx_pre['data'][0][event[0]-150000:event[0]+150000])


# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:


y = nsx_pre['data'][8]


# In[77]:


plt.plot(y)


# In[97]:


from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
from scipy import signal


# In[78]:


ecog_trials = []
for i in event:
    ecog_trials.append(nsx_pre['data'][0][i-50000:i+100000])


# In[270]:


ecog_trials_post = []
for i in event:
    ecog_trials_post.append(nsx_post['data'][0][i-50000:i+100000])


# In[271]:


len(ecog_trials_post)


# In[87]:


y = ecog_trials[0]
Y = fft(y)
# fftshift
shift_Y = fftshift(Y)

# the positive part of fft, get from fft
pos_Y_from_fft = Y[:Y.size//2]

# the positive part of fft, get from shift fft
pos_Y_from_shift = shift_Y[shift_Y.size//2:]

# plot the figures
plt.figure(figsize=(10, 12))

plt.subplot(511)
plt.plot(y)

plt.subplot(512)
plt.plot(np.abs(Y))

plt.subplot(513)
plt.plot(np.abs(shift_Y))

plt.subplot(514)
plt.plot(np.abs(pos_Y_from_fft))

plt.subplot(515)
plt.plot(np.abs(pos_Y_from_shift))
plt.show()


# In[94]:


xx_lim = np.arange(-5,10,1/10000)


# In[223]:


b, a = signal.butter(2, [2*10/fs,2*400/fs], 'bandpass')
yy = signal.filtfilt(b, a, y)


# In[224]:


YY = fft(yy)
subplot(211)
plt.plot(np.abs(Y))
subplot(212)
plt.plot(np.abs(YY))


# In[199]:


yyy = np.delete(yy, argmax(yy)) 


# In[228]:


plt.plot(y)


# In[225]:


nfft = 200
fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.plot(xx_lim, yy)
Pxx, freqs, bins, im = ax2.specgram(yy, NFFT=nfft, Fs=fs, noverlap=100)
# The `specgram` method returns 4 objects. They are:
# - Pxx: the periodogram
# - freqs: the frequency vector
# - bins: the centers of the time bins
# - im: the matplotlib.image.AxesImage instance representing the data in the plot
plt.show()


# In[153]:


def my_specgram(yy, NFFT=nfft, Fs=fs, Fc=0, detrend=mlab.detrend_none,
             window=mlab.window_hanning, noverlap=100,
             cmap=None, xextent=None, pad_to=None, sides='default',
             scale_by_freq=None, minfreq = None, maxfreq = None, **kwargs):
    """
    call signature::

      specgram(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=128,
               cmap=None, xextent=None, pad_to=None, sides='default',
               scale_by_freq=None, minfreq = None, maxfreq = None, **kwargs)

    Compute a spectrogram of data in *x*.  Data are split into
    *NFFT* length segments and the PSD of each section is
    computed.  The windowing function *window* is applied to each
    segment, and the amount of overlap of each segment is
    specified with *noverlap*.

    %(PSD)s

      *Fc*: integer
        The center frequency of *x* (defaults to 0), which offsets
        the y extents of the plot to reflect the frequency range used
        when a signal is acquired and then filtered and downsampled to
        baseband.

      *cmap*:
        A :class:`matplotlib.cm.Colormap` instance; if *None* use
        default determined by rc

      *xextent*:
        The image extent along the x-axis. xextent = (xmin,xmax)
        The default is (0,max(bins)), where bins is the return
        value from :func:`mlab.specgram`

      *minfreq, maxfreq*
        Limits y-axis. Both required

      *kwargs*:

        Additional kwargs are passed on to imshow which makes the
        specgram image

      Return value is (*Pxx*, *freqs*, *bins*, *im*):

      - *bins* are the time points the spectrogram is calculated over
      - *freqs* is an array of frequencies
      - *Pxx* is a len(times) x len(freqs) array of power
      - *im* is a :class:`matplotlib.image.AxesImage` instance

    Note: If *x* is real (i.e. non-complex), only the positive
    spectrum is shown.  If *x* is complex, both positive and
    negative parts of the spectrum are shown.  This can be
    overridden using the *sides* keyword argument.

    **Example:**

    .. plot:: mpl_examples/pylab_examples/specgram_demo.py

    """

    #####################################
    # modified  axes.specgram() to limit
    # the frequencies plotted
    #####################################

    # this will fail if there isn't a current axis in the global scope
    ax = gca()
    Pxx, freqs, bins = mlab.specgram(yy, nfft, fs, detrend,
         window, noverlap, pad_to, sides, scale_by_freq)

    # modified here
    #####################################
    if minfreq is not None and maxfreq is not None:
        Pxx = Pxx[(freqs >= minfreq) & (freqs <= maxfreq)]
        freqs = freqs[(freqs >= minfreq) & (freqs <= maxfreq)]
    #####################################

    Z = 10. * np.log10(Pxx)
    Z = np.flipud(Z)

    if xextent is None: xextent = 0, np.amax(bins)
    xmin, xmax = xextent
    freqs += Fc
    extent = xmin, xmax, freqs[0], freqs[-1]
    im = ax.imshow(Z, cmap, extent=extent, **kwargs)
    ax.axis('auto')

    return Pxx, freqs, bins, im


# In[207]:


Pxx, freqs, bins, im = my_specgram(yy, NFFT=nfft, Fs=fs, noverlap=100, 
                                cmap=plt.cm.gist_heat, minfreq = 4, maxfreq = 200,xextent = (-5,10))
plt.show()


# In[279]:


psd = plt.figure(figsize=(35,50))
for i in range(len(ecog_trials)):
    plt.subplot(10,6,i+1)
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / s')
    plt.ylabel('Freq / Hz')
    plt.specgram(ecog_trials[i],NFFT=256, Fs=10000, noverlap=128, xextent = (-5,10))
    plt.colorbar()
psd.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/ECOGPSD/PSD.png')


# In[272]:


psd = plt.figure(figsize=(35,100))
for i in range(len(ecog_trials_post)):
    plt.subplot(20,6,i+1)
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / s')
    plt.ylabel('Freq / Hz')
    plt.specgram(ecog_trials_post[i],NFFT=256, Fs=10000, noverlap=128, xextent = (-5,10))
    plt.colorbar()
psd.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/ECOGPSD/PSD_post.png')


# In[288]:


for i in range(len(ecog_trials)):
    fig2 = plt.figure(figsize=(20,15))
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / s')
    plt.ylabel('Freq / Hz')
    plt.specgram(ecog_trials[i],NFFT=256, Fs=10000, noverlap=128, xextent = (-5,10))
    plt.colorbar()
    fig2.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/ECOGPSD/Trials_'+str(i+1)+'.png')


# In[289]:


for i in range(len(ecog_trials_post)):
    fig2 = plt.figure(figsize=(20,15))
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / s')
    plt.ylabel('Freq / Hz')
    plt.specgram(ecog_trials_post[i],NFFT=256, Fs=10000, noverlap=128, xextent = (-5,10))
    plt.colorbar()
    fig2.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/ECOGPSD/Trials_'+str(i+1)+'_post.png')


# In[290]:


for i in range(len(ecog_trials)):
    fig2 = plt.figure(figsize=(20,15))
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / s')
    plt.ylabel('Freq / Hz')
    my_specgram(ecog_trials[i], NFFT=256, Fs=fs, noverlap=128, minfreq = 3, maxfreq = 200,xextent = (-5,10))
    fig2.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/ECOGPSD/Trials_'+str(i+1)+'_slice.png')


# In[291]:


for i in range(len(ecog_trials_post)):
    fig2 = plt.figure(figsize=(20,15))
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / s')
    plt.ylabel('Freq / Hz')
    my_specgram(ecog_trials_post[i], NFFT=256, Fs=fs, noverlap=128, minfreq = 3, maxfreq = 200,xextent = (-5,10))
    fig2.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/ECOGPSD/Trials_'+str(i+1)+'_slice_post.png')


# In[206]:


plt.specgram(yy, NFFT=256, Fs=fs, noverlap=128,cmap=plt.cm.gist_heat,xextent = (-5,10))
#plt.ylim([3,200])
db = plt.colorbar()
plt.show()


# In[112]:


ts = yy
def get_xn(Xs,n):
    '''
    calculate the Fourier coefficient X_n of 
    Discrete Fourier Transform (DFT)
    '''
    L  = len(Xs)
    ks = np.arange(0,L,1)
    xn = np.sum(Xs*np.exp(((-1)*1j*2*np.pi*ks*n)/L))
    return(xn)



# In[113]:


def get_xns(ts):
    '''
    Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
    and multiply the absolute value of the Fourier coefficients by 2, 
    to account for the symetry of the Fourier coefficients above the Nyquest Limit. 
    '''
    mag = []
    L = len(ts)
    for n in range(int(L/2)): # Nyquest Limit
        mag.append(np.abs(get_xn(ts,n))*2)
    return(mag)


# In[115]:


def create_spectrogram(ts,NFFT,noverlap = None):
    '''
          ts: original time series
        NFFT: The number of data points used in each block for the DFT.
          Fs: the number of points sampled per second, so called sample_rate
    noverlap: The number of points of overlap between blocks. The default value is 128. 
    '''
    if noverlap is None:
        noverlap = NFFT/2
    noverlap = int(noverlap)
    starts  = np.arange(0,len(ts),NFFT-noverlap,dtype=int)
    # remove any window with less than NFFT sample size
    starts  = starts[starts + NFFT < len(ts)]
    xns = []
    for start in starts:
        # short term discrete fourier transform
        ts_window = get_xns(ts[start:start + NFFT]) 
        xns.append(ts_window)
    specX = np.array(xns).T
    # rescale the absolute value of the spectrogram as rescaling is standard
    spec = 10*np.log10(specX)
    assert spec.shape[1] == len(starts) 
    return(starts,spec)

L = 200
noverlap = 100
ts = yy
starts, spec = create_spectrogram(ts,L,noverlap = noverlap )


# In[116]:


def plot_spectrogram(spec,ks,sample_rate, L, starts, mappable = None):
    plt.figure(figsize=(20,8))
    plt_spec = plt.imshow(spec,origin='lower')


    plt.title("Spectrogram L={} Spectrogram.shape={}".format(L,spec.shape))
    plt.colorbar(mappable,use_gridspec=True)
    plt.show()
    return(plt_spec)


# In[238]:


fig = plt.figure(figsize=(35,50))
for i in range(len(trials)):
    plt.subplot(10,6,i+1)
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / ms')
    plt.ylabel('Voltage / uV')
    plt.ylim((-1000, 2000))
    plt.plot(x_lim,trials[i])
    ave_trials += np.array(trials[i])
ave_trials /= len(trials)
fig.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/Trials.png')


# In[237]:


ave_trials = np.zeros(shape=len(trials[0]))
for i in range(len(trials)):
    ave_trials += np.array(trials[i])
ave_trials /= len(trials)
im = plt.figure(figsize=(20,15))
plt.title('TrialAverage')
plt.xlabel('Time / ms')
plt.ylabel('Voltage / uV')
plt.ylim((-1000, 2000))
plt.plot(x_lim,ave_trials)
im.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/Trials_Ave.png')


# In[236]:


for i in range(len(trials)):
    fig2 = plt.figure(figsize=(20,15))
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / ms')
    plt.ylabel('Voltage / uV')
    plt.ylim((-1000, 2000))
    plt.plot(x_lim,trials[i])
    fig2.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/Trials_'+str(i+1)+'.png')


# In[325]:


fig,ax = plt.subplots()
ax.psd(ecog_trials[0], NFFT=256, Fs=fs, noverlap=128,label='pre')
ax.psd(ecog_trials_post[0], NFFT=256, Fs=fs, noverlap=128,label='post')
logfmt = matplotlib.ticker.LogFormatterExponent(base=200.0, labelOnlyBase=True)
ax.xaxis.set_major_formatter(logfmt)
plt.xlim([4,200])
ax.legend()


# In[347]:


ave_trials = np.zeros(shape=len(trials[0]))
for i in range(len(trials)):
    ave_trials += np.array(trials[i])
ave_trials /= len(trials)

ave_trials_post = np.zeros(shape=len(trials_post[0]))
for i in range(len(trials_post)):
    ave_trials_post += np.array(trials_post[i])
ave_trials_post /= len(trials_post)


# In[348]:


mean(ave_trials)-mean(ave_trials_post)


# In[349]:


import pandas as pd


# In[350]:


df=pd.DataFrame()


# In[351]:


df["pre_rTMS"]=ave_trials

df["post_rTMS"]=ave_trials_post


# In[353]:


plt.boxplot(x=df.values,labels=df.columns)


# In[354]:


plt.boxplot(ave_trials)


# In[372]:


fig = plt.figure(figsize=(35,50))
for i in range(len(ecog_trials)):
    plt.subplot(10,6,i+1)
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / min')
    plt.ylabel('Voltage / uV')
    plt.plot(xx_lim,ecog_trials[i])
fig.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/ECOG/Trials.png')


# In[361]:


len(ecog_trials[0])


# In[377]:


fig = plt.figure(figsize=(35,50))
for i in range(len(ecog_trials)):
    plt.subplot(10,6,i+1)
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / ms')
    plt.ylabel('Voltage / uV')
    plt.plot(np.arange(-30,50),ecog_trials[3][50000-30:50000+50])
fig.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/ECOG/Trials_zoom.png')


# In[375]:


fig = plt.figure(figsize=(35,100))
for i in range(len(ecog_trials_post)):
    plt.subplot(20,6,i+1)
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / min')
    plt.ylabel('Voltage / uV')
    plt.plot(xx_lim,ecog_trials_post[i])
fig.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/ECOG/Trials_zpost.png')


# In[378]:


fig = plt.figure(figsize=(35,100))
for i in range(len(ecog_trials_post)):
    plt.subplot(20,6,i+1)
    plt.title('Trial:'+str(i+1))
    plt.xlabel('Time / ms')
    plt.ylabel('Voltage / uV')
    plt.plot(np.arange(-30,50),ecog_trials_post[3][50000-30:50000+50])
fig.savefig('/home/kenichirotsutsui/Documents/Zhao/demo/ECOG/Trials_zpost_zoom.png')


# In[ ]:




