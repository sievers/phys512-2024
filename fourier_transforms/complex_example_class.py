import numpy as np

a=1+2J

#let's check some exponential sums to show they go to zero
k=0
N=28
x=np.arange(N)
vec=np.exp(2J*np.pi*k*x/N)
print('sum(vector) is :',np.sum(vec))

