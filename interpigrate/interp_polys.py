import numpy as np
from matplotlib import pyplot as plt

ord=15
x=np.linspace(-1,1,ord)
k=5
xx=np.hstack([x[:k],x[k+1:]])
norm=np.prod(x[k]-xx)
xfine=np.linspace(-1,1,1001)
y=1.0
for i in range(len(xx)): #loop over all points that should be zero
    y=y*(xfine-xx[i]) #multiply x-xi
y=y/norm

plt.ion()
plt.clf()
plt.plot(xfine,y)
plt.plot(xx,0*xx,'*')
plt.plot(x[k],1,'o')
plt.show()
