import numpy as np

def num_deriv(fun,x,dx=0.01,eps=1e-16):

    xx=x+np.linspace(-1,1,5)*dx
    ff=fun(xx)
    pp=np.polyfit(xx-x,ff,4)
    f3=pp[3]
    f0=pp[0]
    mydx=(eps*f0/f3)**(1/3)

    xl=x+mydx
    xr=x-mydx
    dx2=xr-xl
    fr=fun(xr)
    fl=fun(xl)
    return (fr-fl)/(dx2),mydx




    
