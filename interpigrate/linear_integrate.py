import numpy as np

fun=np.cos

x=np.linspace(0,np.pi/2,40)
y=fun(x)
dx=x[1]-x[0]
ans=(y[0]+y[-1]+2*np.sum(y[1:-1]))/2*dx
truth=1.0
print('Integral is ',ans,' with error ',ans-truth)
