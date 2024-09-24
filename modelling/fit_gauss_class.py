import numpy as np

def gauss_derivs(p,x):
    amp=p[0]
    x0=p[1]
    var=p[2]
    y=amp*np.exp(-0.5*(x-x0)**2/var)

    A=np.zeros([len(x),len(p)])
    A[:,0]=np.exp(-0.5*(x-x0)**2/var)
    A[:,1]=y*(x-x0)/var
    A[:,2]=y*(x-x0)**2/var**2/2

    return y,A
    
x=np.linspace(-10,10,1001)
x0=0.5
sig=1.5
y_true=np.exp(-0.5*(x-x0)**2/sig**2)
n=0.2
y=y_true+np.random.randn(len(x))*n

#amp=p[0], x0=p[1],var=p[2]
p_guess=np.asarray([1,0,1.0])
p_cur=p_guess.copy()
plt.clf()
plt.plot(x,y,'.')
yy,A=gauss_derivs(p_guess,x)
plt.plot(x,yy)
N=np.eye(len(x))*n**2
Ninv=np.linalg.inv(N)
for i in range(10):    
    y_pred,A=gauss_derivs(p_cur,x)
    r=y-y_pred
    #print('current residual: ',np.sum(r**2)/n**2)
    lhs=A.T@Ninv@A
    rhs=A.T@Ninv@r
    p_cur=p_cur+np.linalg.inv(lhs)@rhs

plt.plot(x,y_pred)
par_cov=np.linalg.inv(lhs)
par_errs=np.sqrt(np.diag(par_cov))
print("par errs: ",par_errs)
print("actual errs: ",p_cur-np.asarray([1,x0,sig**2]))
