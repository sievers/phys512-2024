import numpy as np
import time
import ctypes


mylib=ctypes.cdll.LoadLibrary("libhist2d_c.so")
hist2d_c=mylib.hist2d

hist2d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long,ctypes.c_long,ctypes.c_long]
#void hist2d(long *inds, double *grid, long n, long nx, long ny)

def hist_2d_wrapper(xy,grid):
    n=xy.shape[0]
    nx=grid.shape[1]
    ny=grid.shape[0]
    inds=np.asarray(np.round(xy),dtype='int')
    hist2d_c(inds.ctypes.data,grid.ctypes.data,n,nx,ny)


def hist_2d(xy,grid):
    ixy=np.asarray(np.round(xy),dtype='int')
    n=xy.shape[0]
    for i in range(n):
        grid[ixy[i,0],ixy[i,1]]=grid[ixy[i,0],ixy[i,1]]+1

npix=1000
npt=2000000
ndim=2
xy=np.random.rand(npt,ndim)*(npix-1)  #-1 is just to make sure we don't go off the edge

grid=np.zeros([npix,npix])
grid2=np.zeros([npix,npix])

#ipix=np.asarray(np.round(xy),dtype='int')
t1=time.time()
hist_2d(xy,grid)
#for i in range(npt):
#    grid[ipix[i,0],ipix[i,1]]=grid[ipix[i,0],ipix[i,1]]+1
t2=time.time()
print('time per particle to project was ' + repr((t2-t1)/npt))

t1=time.time()
hist_2d_wrapper(xy,grid2)
t2=time.time()
print('time per particle to project in C was ' + repr((t2-t1)/npt))
print('std is ',np.std(grid-grid2))
