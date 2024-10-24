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

cloud_amps=np.abs(cloudft)
cloud_phase=np.angle(cloudft)
arts_amps=np.abs(artsft)
arts_phases=np.angle(artsft)

cphases=np.exp(1J*cloud_phase)#*arts_amps
aphases=cloud_amps*np.exp(1J*arts_phases)

cphase_image=np.fft.irfft2(cphases)
aphase_image=np.fft.irfft2(aphases)
plt.figure(1)
plt.clf()
plt.imshow(cphase_image,cmap='gray',vmin=-0.002,vmax=0.005)
plt.title('cloudberry phase, arts amp')
plt.figure(2)
plt.clf()
plt.imshow(aphase_image,cmap='gray')
plt.title('arts phases, cloudberry amps')
plt.show()


