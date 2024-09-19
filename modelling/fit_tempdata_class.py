import numpy as np
from openpyxl import load_workbook
from matplotlib import pyplot as plt


crud=load_workbook('A02_GAIN_MIN20_373.xlsx')
sheet=crud['All']

nu=[sheet['A'+repr(i+1)].value for i in range(4096)]
nu=np.asarray(nu)
gain=[sheet['B'+repr(i+1)].value for i in range(4096)]
gain=np.asarray(gain)
nu=nu/1e6 #convert frequency from Hz to MHz

nu_min=15

ii=nu>nu_min
nu=nu[ii]
gain=gain[ii]

#at this point, we have frequencies and gains in [nu,gain]
plt.ion()
plt.clf()
plt.plot(nu,gain)
plt.show()

nu_scale=nu-nu.min()
nu_scale=nu_scale/nu_scale.max()
nu_scale=2*(nu_scale)-1

order=5

A=np.polynomial.legendre.legvander(nu_scale,order)
lhs=A.T@A
rhs=A.T@gain
fitp=np.linalg.inv(lhs)@rhs
gpred=A@fitp
plt.plot(nu,gpred)

resid=(gain-gpred)
sig=np.std(resid)
print('residual scatter is ',sig)
N=np.eye(len(gain))*sig**2

lhs=A.T@np.linalg.inv(N)@A
errs=np.sqrt(np.diag(np.linalg.inv(lhs)))
print('parameter values: ',fitp)
print('parameter errors: ',errs)

var_data=A@np.linalg.inv(lhs)@A.T
err_data=np.sqrt(np.diag(var_data))
print('mean err in prediction: ',np.mean(err_data))
