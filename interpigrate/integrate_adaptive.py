import numpy as np

def myexp(x):
    return 1+np.exp(-0.5*x**2/0.1**2)

def integrate(fun,a,b,tol=1e-6):
    vec=np.linspace(a,b,5)
    vals=fun(vec)
    dx=(b-a)/4

    a1=(vals[0]+4*vals[2]+vals[4])/3*2*dx
    a2=(vals[0]+4*vals[1]+2*vals[2]+4*vals[3]+vals[4])/3*dx

    
    if np.abs(a2-a1)<tol:
        return a2
    else:
        mid=(a+b)/2
        a1=integrate(fun,a,mid,tol/2)
        a2=integrate(fun,mid,b,tol/2)
        return a1+a2
