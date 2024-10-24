import numpy as np
from matplotlib import pyplot as plt
plt.ion()
myim=plt.imread('meerkat_small.png')

red=myim[:,:,0] #this is the red part of the rgb image
myft=np.fft.rfft2(red)
plt.figure(1)
plt.clf()
plt.imshow(red,cmap='gray')
plt.show()

plt.figure(2)
plt.clf()
plt.imshow(np.log10(np.abs(myft)),vmin=1,vmax=3)
plt.axis('auto')
plt.colorbar()
plt.show()
