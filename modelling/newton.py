import numpy
from matplotlib import pyplot as plt

def calc_lorentz(p,t):
    y=p[0]/(p[1]+(t-p[2])**2)
    grad=numpy.zeros([t.size,p.size])
    #now differentiate w.r.t. all the parameters
    grad[:,0]=1.0/(p[1]+(t-p[2])**2)
    grad[:,1]=-p[0]/(p[1]+(t-p[2])**2)**2
    grad[:,2]=p[0]*2*(t-p[2])/(p[1]+(t-p[2])**2)**2
    return y,grad

t=numpy.arange(-5,5,0.1)
p_true=numpy.array([3,3,-0.5])
x_true,grad=calc_lorentz(p_true,t)
x=x_true+0.1*numpy.random.randn(t.size)
plt.ion()
plt.clf()
plt.plot(t,x_true)
plt.plot(t,x,'.')
p0=numpy.array([2,3,-0.4]) #starting guess, close but not exact

p=p0.copy()
for j in range(5):
    pred,grad=calc_lorentz(p,t)
    r=x-pred
    err=(r**2).sum()
    r=numpy.matrix(r).transpose()
    grad=numpy.matrix(grad)

    lhs=grad.transpose()*grad
    rhs=grad.transpose()*r
    dp=numpy.linalg.inv(lhs)*(rhs)
    for jj in range(p.size):
        p[jj]=p[jj]+dp[jj]
    print p,err
plt.plot(t,pred)
plt.savefig('lorentz_fit.png')
#2.0/(3.0+(t-0.5)**2) + 0.2*numpy.random.randn(t.size)

