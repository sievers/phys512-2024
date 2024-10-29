import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=10000
x=np.arange(n)
y_true=1/np.random.rand(n)**1.5  #this is a quick, hacky way to make some spikes
y_true=y_true/y_true.max() #let's make the largest peak 1
tau=5 #instrument response time, in samples
kernel=np.exp(-x/tau)  #let's say instrument response is exponential decay
#the noiseless signal we see will be the convolution of the true signal
#with the instrument
y_true_conv=np.fft.irfft(np.fft.rfft(y_true)*np.fft.rfft(kernel))

#let's add some Gaussian noise with standard deviation sig
sig=0.02
y=y_true_conv+np.random.randn(n)*sig

#and let's make the matched filter.  Since the noise is white,
#we know that N^-1 is 1/sig**2
lhs=kernel@kernel/sig**2
yft=np.fft.rfft(y)
kft=np.fft.rfft(kernel)
rhs=np.fft.irfft(yft*np.conj(kft))/sig**2
amp=rhs/lhs  #this is the properly normalized matched filter
err=1/np.sqrt(lhs) #this is our formal error on the matched filter fits

print('error in max amplitude is ',amp.max()-y_true.max(),' analytic: ',err)
