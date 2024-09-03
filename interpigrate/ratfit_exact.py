import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q


#1*p0 + x*p1 +x**2+p2+... -q1*x - q2*x**2... = y

n=4
m=7
x=np.linspace(-3,3,n+m-1)
y=np.exp(-0.5*x**2)
p,q=rat_fit(x,y,n,m)
xx=np.linspace(5*x[0],5*x[-1],1001)
y_true=np.exp(-0.5*xx**2)
pred=rat_eval(p,q,xx)
plt.clf();plt.plot(x,y,'*')
plt.plot(xx,y_true)
plt.plot(xx,pred)
plt.title('Rational Function Fit to Gaussian')
plt.savefig('ratfit_gauss.png')

fitp=np.polyfit(x,y,n+m-1)
pred_poly=np.polyval(fitp,xx)

plt.clf();plt.plot(x,y,'*');plt.plot(xx,y_true);plt.plot(xx,pred)

assert(1==0)

xx=np.linspace(-2,2,1001)
yy=np.exp(-0.5*xx**2)
yy_interp=rat_eval(p,q,xx)
plt.ion()
plt.clf()
plt.plot(xx,yy_interp-yy)


#we can use numpy's polynomial fitter to see how that does
pp=np.polyfit(x,y,n+m) #use same number of terms
yy_poly=np.polyval(pp,xx)
plt.plot(xx,yy_poly-yy)
plt.legend(['Ratfun err','Poly err'])
plt.savefig('ratfit_vs_poly.png')
