import numpy as np
from matplotlib import pyplot as plt

plt.ion()
N=1000000
alpha=1.1
x=np.random.rand(N)
s=x**(1/(1-alpha))

aa,bb=np.histogram(s,np.linspace(1,21,401))
b=(bb[1:]+bb[:-1])/2
pred=b**-alpha
pred=pred/pred.sum()
aa=aa/aa.sum()
plt.clf()
plt.plot(b,aa,'.')
plt.plot(b,pred,'r');
plt.legend(['Measured PDF','Prediction'])
plt.show()
plt.savefig('powlaw_pdf.png')
