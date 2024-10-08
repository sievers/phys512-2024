import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.linspace(-5,5,1024)
y=np.exp(-0.5*x**2/(0.1**2))

yft=np.fft.fft(y)
plt.figure(1)
plt.clf()
plt.plot(x,y)
plt.show()
plt.figure(2)
plt.clf()
plt.plot(np.abs(yft))
plt.show()

yft_shift=np.fft.fftshift(yft)
plt.plot(np.abs(yft_shift))
