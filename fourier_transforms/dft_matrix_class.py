import numpy as np

N=32
A=np.zeros([N,N],dtype='complex')
for i in range(N):
    for j in range(N):
        A[i,j]=np.exp(-2J*np.pi*i*j/N)
f=np.ones(N)
F=A@f

Ainv=A.conj()/N

print('inverse check: ',np.sum(np.abs(Ainv@A-np.eye(N))))

f=np.random.randn(N)
F=A@f
myft=np.fft.fft(f)
print('error between matrix and numpy function: ',np.std(myft-F))
