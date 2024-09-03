import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

x=np.linspace(-1,1,11)
y=np.exp(x)

y[-2:]=y[-3]

spln=CubicSpline(x,y)

xx=np.linspace(-1,1,1001)
yy=spln(xx)
yy_true=np.exp(xx)

plt.clf()
plt.plot(x,y,'.')
plt.plot(xx,yy)
#plt.plot(xx,yy_true)
plt.show()
