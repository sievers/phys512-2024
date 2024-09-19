import numpy as np
from matplotlib import pyplot as plt
plt.ion()

t=np.linspace(-1,1,30001)
n=len(t)

c0=3
c1=5

x=c0+c1*t+np.random.randn(n)

plt.clf()
plt.plot(t,x,'.')
plt.show()

npoly=35
A=np.zeros([n,npoly])
for i in range(npoly):
    A[:,i]=t**i

u,s,v=np.linalg.svd(A,0)
print('condition number of regular polynomials is ',s.max()/s.min())
    
#Al=np.zeros([n,npoly])
#Al[:,0]=1
#Al[:,1]=t
#for i in range(1,npoly-1):
#    Al[:,i+1]=((2*i+1)*t*Al[:,i]-i*Al[:,i-1])/(i+1)

A=np.polynomial.chebyshev.chebvander(t,npoly)
u,s,v=np.linalg.svd(A,0)
print('condition number of chebyshev polynomials is ',s.max()/s.min())
lhs=A.T@A  #there would be an N if we had interesting noise
rhs=A.T@x
m=np.linalg.pinv(lhs)@rhs
#print('maximum likelihood parameters are: ',m)

u,s,v=np.linalg.svd(A,0)
m_new=v.T@np.diag(1/s)@u.T@x
pred_new=A@m_new

pred=A@m
plt.plot(t,pred)
plt.plot(t,pred_new)
