import numpy as np
from matplotlib import pyplot as plt

plt.ion()

cloud=plt.imread('cloudberry.png')
arts=plt.imread('mcgill_arts.jpeg')


#do some resizing/cropping to make the images the same size
arts=arts[::2,::2,0]
cloud=cloud[:,:,0]
i0=(arts.shape[0]-cloud.shape[0])//2
i1=(arts.shape[1]-cloud.shape[1])//2
arts=arts[i0:i0+cloud.shape[0],i1:i1+cloud.shape[1]]

cloudft=np.fft.rfft2(cloud)
artsft=np.fft.rfft2(arts)

cloud_phase=np.abs(artsft)*np.exp(1j*np.angle(cloudft))
arts_phase=np.abs(cloudft)*np.exp(1j*np.angle(artsft))


cc=np.fft.irfft2(cloud_phase)
aa=np.fft.irfft2(arts_phase)

