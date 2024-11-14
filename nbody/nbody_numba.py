import numpy as np
from matplotlib import pyplot as plt
import numba as nb
plt.ion()

@nb.njit(parallel=True)
def calc_acc_nb(a,x,m,soft):
    a[:]=0
    n=len(m)
    ndim=x.shape[1]
    print('ndim and n are ',ndim,n)
    dx=np.zeros(ndim,dtype='float')
    for i in nb.prange(n):
        for j in range(n):
            rsqr=soft**2
            for k in range(ndim):
                dx[k]=x[i,k]-x[j,k]
                rsqr=rsqr+dx[k]**2
            r=np.sqrt(rsqr)
            r3=1/(r*rsqr)
            for k in range(ndim):
                a[i,k]=a[i,k]-m[j]*dx[k]*r3

class Particles:
    def __init__(self,x,v,m=1.0,soft=0.01):
        self.x=x
        self.v=v
        self.n=x.shape[0]
        self.ndim=x.shape[1]
        self.a=np.empty([self.n,self.ndim])
        self.m=np.empty(self.n)
        self.m[:]=m
        self.soft=soft
    def calc_acc_old(self):
        self.a[:]=0
        for i in range(self.n):
            for j in range(self.n):
                if not(i==j):
                    dx=self.x[i,:]-self.x[j,:]
                    rsqr=np.sum(dx**2)
                    if rsqr<self.soft**2:
                        self.a[i,:]=self.a[i,:]-self.m[j]*dx/(self.soft**3)
                    else:
                        r=np.sqrt(rsqr)
                        self.a[i,:]=self.a[i,:]-self.m[j]*dx/(rsqr*r)        
    def calc_acc(self):
        if True:
            calc_acc_nb(self.a,self.x,self.m,self.soft)
            return
        self.a[:]=0
        for i in range(self.n):
            dx=self.x[i,:]-self.x
            rsqr=np.sum(dx**2,axis=1)+self.soft**2
            r=np.sqrt(rsqr)
            for j in range(dx.shape[1]):
                dx[:,j]=dx[:,j]*self.m
            for j in range(dx.shape[1]):
                self.a[i,j]=-np.sum(dx[:,j]/(rsqr*r))
                    
    def update(self,dt):
        self.calc_acc()
        self.x=self.x+dt*self.v
        self.v=self.v+dt*self.a

    def plot(self,clear=True,lims=None):
        if clear:
            plt.clf()
        plt.plot(self.x[:,0],self.x[:,1],'.')
        if not(lims is None):
            plt.axis(lims)
        plt.pause(0.0001)
        


n=10000
        
#x=np.zeros([n,2])
x=np.random.randn(n,2)
v=np.zeros([n,2])
m=np.ones(n)/n
#x[0,0]=1
#x[1,0]=-1
#v[0,1]=1
#v[1,1]=-1



lims=np.asarray([-1.5,1.5,-1.5,1.5])
parts=Particles(x,v*0,m,soft=0.1)
parts.plot(lims=lims*2,clear=True)
dt=0.01
oversamp=1
for i in range(10000):
    for j in range(oversamp):
        parts.update(dt/oversamp)
    parts.plot(clear=True)
    print(parts.x)
