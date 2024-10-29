import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
plt.ion()

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
