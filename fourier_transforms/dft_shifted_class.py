import numpy as np
from matplotlib import pyplot as plt
plt.ion()

N=1024
x=np.linspace(-5,5,N)
y=np.exp(-0.5*(x-2)**2/(0.1**2))
dx_samp=x[1]-x[0]

xshift=-1
dx=int(xshift/dx_samp)

kvec=np.arange(N)/N
phase=2*np.pi*kvec*dx
shift_vec=np.exp(-1J*phase)
yft=np.fft.fft(y)
yft_shifted=yft*shift_vec
y_back=np.fft.ifft(yft_shifted)
y_real=np.sum(np.abs(np.real(y_back)))
y_im=np.sum(np.abs(np.imag(y_back)))
print('real/im parts of  y_back are ',y_real,y_im)
y_back=np.real(y_back)

plt.clf()
plt.plot(x,y)
plt.plot(x,y_back)
plt.show()
