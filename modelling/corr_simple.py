import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=10000
x=np.random.randn(n)
y=np.random.randn(n)
a=x+y
b=x-y

aa=(x/3+3*y)
bb=(x/3-3*y)
plt.clf()
plt.plot(aa,bb,'.')
plt.plot(a,b,'.')
plt.legend([r'$a,b=x/3\pm 3y$',r'$a,b=x\pm y$'])
plt.xlabel('a')
plt.ylabel('b')
plt.title('Correlated/Uncorrelated Noise')
plt.show()
plt.savefig('corrnoise_linear.png')
