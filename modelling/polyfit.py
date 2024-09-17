import numpy as np
from matplotlib import pyplot as plt
plt.ion()


t=np.linspace(-5,5,101)
x_true=t**3-0.5*t**2
x=x_true+10*np.random.randn(len(t))

ord=5
A=np.zeros([len(t),ord+1])
for i in range(ord+1):
    A[:,i]=t**i

lhs=A.T@A
rhs=A.T@x
fitp=np.linalg.inv(lhs)@rhs
pred=A@fitp
plt.clf()
plt.plot(t,x,'*')
plt.plot(t,pred)
plt.show()
print('squared residual is ',np.sum((x-pred)**2))
