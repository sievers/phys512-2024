import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.linspace(-5,5,1024)
y=np.exp(-0.5*(x-2)**2/(0.1**2))

yft=np.fft.fft(y)
y_back=np.real(np.fft.ifft(yft))

plt.clf()
plt.plot(x,y)
plt.plot(x,y_back)
plt.show()

yft_flipped=np.conj(yft)
y_back_flipped=np.fft.ifft(yft_flipped)
plt.plot(x,np.real(y_back_flipped))
