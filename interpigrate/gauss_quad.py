import numpy as np

def lorentz(x):
    return 1/(1+x**2)

def simpson(fun,a,b,npt):
    if npt%2==0:
        npt=npt+1
    x=np.linspace(a,b,npt)
    dx=x[1]-x[0]
    y=fun(x)

    val=y[0]+y[-1]+4*np.sum(y[1:-1:2])+2*np.sum(y[2:-1:2])
    return val*dx/3
def gauss3(fun,a,b,npt):
    """Do 3-point Gaussian quadrature with roughly npt function evaluations"""
    nn=npt//3
    dx=(b-a)/(nn)
    x0=a+dx/2+np.arange(nn)*dx
    xl=x0-np.sqrt(3/5)*dx/2
    xr=x0+np.sqrt(3/5)*dx/2

    vals=fun(x0)*8+fun(xl)*5+fun(xr)*5
    return np.sum(vals)/9*dx/2

def gauss5(fun,a,b,npt):
    """Do 5-point Gaussian quadrature with roughly npt function evaluations"""
    nn=npt//5
    dx=(b-a)/(nn)
    d1=np.sqrt(5-2*np.sqrt(10/7))/3
    d2=np.sqrt(5+2*np.sqrt(10/7))/3
    print(d1,d2,nn*5)
    x0=a+dx/2+np.arange(nn)*dx
    xl1=x0-d1*dx/2
    xl2=x0-d2*dx/2
    xr1=x0+d1*dx/2
    xr2=x0+d2*dx/2
    y0=fun(x0)
    y1=fun(xl1)+fun(xr1)
    y2=fun(xl2)+fun(xr2)

    w0=128/225
    w1=(322+13*np.sqrt(70))/900
    w2=(322-13*np.sqrt(70))/900
    print(w0,w1,w2)

    vals=y0*w0+y1*w1+y2*w2
    return np.sum(vals)*dx/2



a=-2
b=2

if True:
    fun=np.exp
    ans=np.exp(b)-np.exp(a)
else:
    fun=lorentz
    ans=np.arctan(b)-np.arctan(a)

ys=simpson(fun,a,b,15)
y3=gauss3(fun,a,b,15)
y5=gauss5(fun,a,b,15)

print('simpson error: ',ys-ans)
print('gauss3 error: ',y3-ans)
print('gauss5 error: ',y5-ans)

