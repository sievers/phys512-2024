import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
plt.ion()

def f(x,y):
    dydx=np.zeros(3)
    tau1=1
    tau2=1e-5
    dydx[0]=-y[0]/tau1
    dydx[1]=y[0]/tau1-y[1]/tau2
    dydx[2]=y[1]/tau2
    return dydx

y0=np.asarray([1.0,0,0])
x0=0
x1=1
ans_rk4=integrate.solve_ivp(f,[x0,x1],y0)
print('rk4 took ',ans_rk4.nfev,' function evaluations.')
ans_radau=integrate.solve_ivp(f,[x0,x1],y0,method='Radau')
print('radau took ',ans_radau.nfev,' function evalutions')
plt.clf()
plt.plot(ans_rk4.t,ans_rk4.y[0,:],'.')
plt.plot(ans_radau.t,ans_radau.y[0,:],'.')
plt.show()

