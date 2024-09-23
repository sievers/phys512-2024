import numpy as np
import ratfit_exact
from scipy.interpolate import CubicSpline
def lorentz(x):
    return 1/(1+x**2)

funs=[np.cos,lorentz]
x=np.linspace(-np.pi,np.pi,9)
xfine=np.linspace(-np.pi,np.pi,1001)
for fun in funs:
    y=fun(x)
    yfine=fun(xfine)
    fitp=np.polyfit(x,y,len(x)-1)
    print('polynomial error: ',np.std(np.polyval(fitp,xfine)-yfine))
    spln=CubicSpline(x,y)
    pred=spln(xfine)
    print('Spline error: ',np.std(pred-yfine))
    #n=len(x)//2
    n=1
    m=len(x)-n+1
    fitp=ratfit_exact.rat_fit(x,y,n,m)
    pred=ratfit_exact.rat_eval(fitp[0],fitp[1],xfine)
    print('rat err: ',np.std(pred-yfine))
