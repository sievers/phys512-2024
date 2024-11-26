import numpy as np
import time
import numba as nb


@nb.njit
def project(ixy,grid):
    n=ixy.shape[0]
    for i in range(n):
        grid[ixy[i,0],ixy[i,1]]=grid[ixy[i,0],ixy[i,1]]+1
    

def hist_2d(xy,grid):
    ixy=np.asarray(np.round(xy),dtype='int')
    #n=xy.shape[0]
    project(ixy,grid)
    
npix=1000
npt=2000000
ndim=2
xy=np.random.rand(npt,ndim)*(npix-1)  #-1 is just to make sure we don't go off the edge

grid=np.zeros([npix,npix])


#ipix=np.asarray(np.round(xy),dtype='int')
t1=time.time()
hist_2d(xy,grid)
#for i in range(npt):
#    grid[ipix[i,0],ipix[i,1]]=grid[ipix[i,0],ipix[i,1]]+1
t2=time.time()
hist_2d(xy,grid)
t3=time.time()

print('time per particle to project was ' + repr((t2-t1)/npt))
print('time per particle on run 2 was ',(t3-t2)/npt)
