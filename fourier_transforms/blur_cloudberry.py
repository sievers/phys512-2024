import numpy as np
from matplotlib import pyplot as plt

def make_chessboard(ncell,npix):
    n=ncell*npix
    mat=np.zeros([n,n])
    for i in range(ncell):
        for j in range(ncell):
            isblack=(i-j)%2==0
            if isblack:
                mat[i*npix:(i+1)*npix,j*npix:(j+1)*npix]=1
    return mat

def get_smooth_kernel(shape,npx):
    nx=shape[0]
    ny=shape[1]
    xvec=np.arange(nx)
    yvec=np.arange(ny)
    xvec[nx//2:]=xvec[nx//2:]-nx
    yvec[ny//2:]=yvec[ny//2:]-ny
    xexp=np.exp(-0.5*xvec**2/npx**2)
    yexp=np.exp(-0.5*yvec**2/npx**2)
    kernel=np.outer(xexp,yexp)
    return kernel/kernel.sum()

def get_kernel_new(sz,npx):
    x=np.arange(sz)
    x[sz//2:]=x[sz//2:]-sz

    xsqr=x**2
    xmat=np.outer(xsqr,np.ones(len(xsqr)))
    rsqr=xmat+xmat.T
    kernel=np.exp(-0.5*rsqr/npx**2)
    kernel=kernel/kernel.sum()
    return kernel
    

plt.ion()
cloud=plt.imread('cloudberry.png')
cloud=cloud[:,:,0]
cloud=cloud[:,180:(180+720)]

sig=3
kernel=get_kernel_new(cloud.shape[0],sig)

cloudft=np.fft.rfft2(cloud)
kernelft=np.fft.rfft2(kernel)
cloud_smooth=np.fft.irfft2(cloudft*kernelft)
plt.figure(1)
plt.clf()
plt.imshow(cloud_smooth)
plt.show()

cloud_smooth_ft=np.fft.rfft2(cloud_smooth)
cloud_back_ft=cloud_smooth_ft/kernelft
cloud_back=np.fft.irfft2(cloud_back_ft)
plt.figure(2)
plt.imshow(cloud_back)
plt.show()
