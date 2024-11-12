import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=300
x=np.zeros(n)
x[n//3:(2*n//3)]=1.0

beta=1.0
alpha=1.0
k=10/n*2*np.pi
#x=np.exp(1j*k*np.arange(n))
x_org=x.copy()
for i in range(int(n/alpha)):
    #set up our function with guard cells
    xx=np.zeros(n+2,dtype='complex')
    xx[1:-1]=x
    xx[0]=x[-1]
    xx[-1]=x[0]
    #this is an estimate of the current state.
    #beta=0 is standard, beta=1.0 is Lax, but
    #beta can also be intermediate
    xcur=beta*(xx[2:]+xx[:-2])/2+(1-beta)*xx[1:-1]
    #we'll use the centered derivative that was unstable
    grad=(xx[2:]-xx[:-2])/2
    #grad=(xx[1:-1]-xx[:-2])
    x=xcur-alpha*grad
    plt.clf()
    plt.plot(np.real(x))
    plt.show()
    plt.pause(0.001)
    #x[t+dt]=
    x2=x_org*(np.cos(k)+1J*alpha*np.sin(k))
    #assert(1==0)

#fn+1 = (exp(ikx+1)+exp(ikx-1))/2+a exp(ikx+1)-exp(ikx-1)/2
#=exp(ikx)(exp(ik)+exp(-ik)/2 + a (exp(ik)-exp(-ik))/2)
#= exp(ikx) (cos(k) + iasin(k))
