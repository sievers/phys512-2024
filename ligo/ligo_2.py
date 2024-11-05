import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
plt.ion()

def smooth_vec(vec,npix):
    x=np.fft.fftfreq(len(vec))*len(vec)
    print(x[:5])
    print(x[-5:])
    gauss=np.exp(-0.5*x**2/npix**2)
    gauss=gauss/gauss.sum()
    return np.fft.irfft(np.fft.rfft(vec)*np.fft.rfft(gauss),len(gauss))
    
def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    tp=template[0]
    tx=template[1]
    return tp,tx
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc



fname='H-H1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
hstrain,dt,utc=read_file(fname)

fname='L-L1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
lstrain,dt,utc=read_file(fname)
#dt,utc are the same for both files, so doesn't matter we overwrote


#th,tl=read_template('GW150914_4_template.hdf5')
template_name='GW150914_4_template.hdf5'
tp,tx=read_template(template_name)

t=np.arange(len(hstrain))*dt
hft=np.fft.rfft(hstrain)
lft=np.fft.rfft(lstrain)


vec=np.linspace(-3*np.pi,3*np.pi,len(hstrain))
window=0.5*(np.cos(vec)+1)
window[np.abs(vec)<2*np.pi]=1

hft_win=np.fft.rfft(hstrain*window)

ttot=dt*len(hstrain)
dnu=1/ttot
nuvec=np.arange(len(hft))*dnu

hps_raw=np.abs(hft_win)**2/len(hft)
hps=smooth_vec(hps_raw,10)

#plt.clf()
#plt.loglog(nuvec,np.abs(hft)**2)
#plt.loglog(nuvec,np.abs(hft_win)**2)
#plt.show()

template_ft=np.fft.rfft(tp)
mf=np.fft.irfft(hft_win*np.conj(template_ft)/hps)
plt.clf();plt.plot(mf);plt.show()

tpft=np.fft.rfft(tp*window)
txft=np.fft.rfft(tx*window)
tft_filt=np.zeros([len(tpft),2],dtype='complex')
tft_filt[:,0]=tpft/hps
tft_filt[:,1]=txft/hps

#tp_filt=np.fft.irfft(tpft/hps,len(tp))
#tx_filt=np.fft.irfft(txft/hps,len(tx))
#tfilt=np.zeros([len(tp_filt),2])
#tfilt[:,0]=tp_filt
#tfilt[:,1]=tx_filt
tfilt=np.fft.irfft(tft_filt,axis=0)

t=0*tfilt
t[:,0]=tp*window
t[:,1]=tx*window
lhs=t.T@tfilt
rhs=np.fft.irfft(np.conj(tft_filt)*hft_win[:,np.newaxis],axis=0)
amps=rhs@np.linalg.inv(lhs)
aa=np.sum(amps**2,axis=1)

#(d-Am)T N-1 (d-Am) - d^T N-1 D = -2 m^T (A^T N-1 d) + m^T A^T^N^-1Am
ind=np.argmax(np.abs(aa))
#chi1=-2*amps[ind,:]@rhs[ind,:]
chi1=np.sum(amps*rhs,axis=1)
chi2=np.sum((amps@lhs)*amps,axis=1)
dchi=-2*chi1+chi2
print('minimum chi2 equiv. sigma: ',np.sqrt(-np.min(dchi)))
