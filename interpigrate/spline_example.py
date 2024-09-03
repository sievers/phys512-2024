import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import interpolate
plt.ion()

x=np.linspace(-2,2,11)
y=np.exp(x)
y[-2:]=y[-3]
#periodic would be somethinglike y[-1]=y[0]

#y=0*y;y[-1]=1

xx=np.linspace(x[0],x[-1],2001)
spln=CubicSpline(x,y)
yy=spln(xx)

plt.clf();
plt.plot(x,y,'*')
plt.plot(xx,yy)

myfun=interpolate.interp1d(x,y,'linear')
yy2=myfun(xx)
plt.plot(xx,yy2)
plt.savefig('spline_out.png')

myfun2=interpolate.PchipInterpolator(xx,yy)
yy3=myfun2(xx)
plt.plot(xx,yy3)
plt.show()
