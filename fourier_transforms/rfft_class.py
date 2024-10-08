import numpy as np
x=np.random.randn(20)
xft=np.fft.fft(x)
print(xft[len(x)//2])
xftr=np.fft.rfft(x)
print('lengths are ',len(xft),len(xftr))

xback=np.fft.irfft(xftr)
print('round trip error: ',np.std(x-xback))

xback2=np.fft.ifft(xft)

