import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.linspace(-1,1,10001)

order=10
A=np.zeros([len(x),order+1])
A[:,0]=1
for i in range(order):
    A[:,i+1]=A[:,i]*x

plt.clf()
plt.plot(x,A)
plt.show()

ATA=A.T@A
e,v=np.linalg.eigh(ATA)
print("largest/smallest eigenvalues are ",e.max(),e.min(),e.max()/e.min())
