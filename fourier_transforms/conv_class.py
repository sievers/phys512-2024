import numpy as np
from matplotlib import pyplot as plt

x=np.linspace(0,20,1024)
x0=3
gsig=2
#ygauss=np.exp(-0.5*(x-x0)**2/gsig**2)
ygauss=np.exp(-0.5*x/gsig)
ygauss=ygauss/ygauss.sum()

ybox=0*x
x0_box=8
width_box=1
ybox[np.abs(x-x0_box)<width_box]=1
ybox=ybox/ybox.sum()

plt.clf()
plt.plot(x,ygauss)
plt.plot(x,ybox)
plt.show()

gft=np.fft.rfft(ygauss)
bft=np.fft.rfft(ybox)
conv_ft=gft*bft
yconv=np.fft.irfft(conv_ft)
plt.plot(x,yconv)

#k=0 term is sum(f(x) exp(0))=sum(f(x))
#k=0 term in convolution is sum(f(x))*sum(g(x)) because H[0]=F[0]*G[0]
