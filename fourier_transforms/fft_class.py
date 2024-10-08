import numpy as np

def myfft(x):
    if len(x)==1:
        return x
    xeven=x[::2]
    xodd=x[1::2]
    xft_even=myfft(xeven)
    xft_odd=myfft(xodd)
    k=np.arange(len(xft_even))
    twid=np.exp(-2J*np.pi*k/len(x))
    ft_low=xft_even+twid*xft_odd
    ft_high=xft_even-twid*xft_odd
    return np.hstack([ft_low,ft_high])


f=np.random.randn(16)
myft=myfft(f)
print('error in dft is ',np.sum(np.abs(myft-np.fft.fft(f))))
