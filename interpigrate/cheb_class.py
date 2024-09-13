import numpy as np

def cheb_mat(x,ord):
    A=np.zeros([len(x),ord])
    A[:,0]=1
    A[:,1]=x
    for i in range(1,ord-1):
        A[:,i+1]=2*x*A[:,i]-A[:,i-1]
    return A

x=np.linspace(-1,1,31)
y=np.exp(x)
A=cheb_mat(x,len(x))
fitp=np.linalg.inv(A)@y
y_pred=A@fitp
nuse=7
xx=np.linspace(-1,1,1001)
yy=np.exp(xx)
Ause=cheb_mat(xx,nuse)
yy_pred=Ause@fitp[:nuse]
yy_err=yy_pred-yy
yy_taylor=1.0
factorial=[1,1,2,6,24,120,720,5040]
for i in range(1,nuse):
    yy_taylor=yy_taylor+(xx**i)/factorial[i]
    #yy_taylor=yy_taylor+xx**i/np.product(1,i+1)
err_taylor=yy_taylor-yy
plt.clf()
plt.plot(xx,yy_err)
plt.plot(xx,err_taylor)
plt.show()
