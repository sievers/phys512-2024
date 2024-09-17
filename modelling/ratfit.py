import numpy as np
from matplotlib import pyplot as plt


def fun(x):
    #return x*np.exp(-0.5*x**2)
    return x**3/(0.5+x**2)

def ratfit_exact(x,y,n,m):
    npt=len(x)
    assert(n+m==npt)

    num_mat=np.zeros([npt,n])
    denom_mat=np.zeros([npt,m])
    num_mat[:,0]=1
    for i in range(1,n):
        num_mat[:,i]=x*num_mat[:,i-1]
    denom_mat[:,0]=y*x
    for i in range(1,m):
        denom_mat[:,i]=x*denom_mat[:,i-1]
    mat=np.append(num_mat,-denom_mat,axis=1)

    coeffs=np.dot(np.linalg.inv(mat),y)
    
    coeffs_num=coeffs[:n]
    coeffs_denom=np.append(1,coeffs[n:])
    
    num=np.polyval(np.flipud(coeffs_num),x)
    denom=np.polyval(np.flipud(coeffs_denom),x)
    pred=num/denom

    #print('error in fit is ' + repr(np.std(pred-y)))
    return coeffs_num,coeffs_denom


def ratfit_lsqr(x,y,aa,bb):
    #Use Newton's method to fit least-squares rational functions with aa,bb as starting parameter guesses
    #derivative of rational function w.r.t. numerator parameters are x^n/denom
    #derivative w.r.t. bottom is -top/denom^2*x^m
    npt=len(x)
    n=len(aa)
    m=len(bb)-1
    bb=bb[1:]
    for i in range(10):
        num=np.polyval(np.flipud(aa),x)
        denom=1+x*np.polyval(np.flipud(bb),x)
        num_mat=np.zeros([npt,n])
        denom_mat=np.zeros([npt,m])
        pred=num/denom
        vec=-pred/denom

        denom_mat[:,0]=vec*x
        for i in range(1,m):
            denom_mat[:,i]=denom_mat[:,i-1]*x
        num_mat[:,0]=1.0/denom
        for i in range(1,n):
            num_mat[:,i]=num_mat[:,i-1]*x
        mat=np.append(num_mat,denom_mat,axis=1)
        lhs=np.dot(mat.transpose(),mat)
        resid=y-pred
        print('residual squared sum is '+repr(np.sum(resid**2)))
        rhs=np.dot(mat.transpose(),resid)
        dpar=np.dot(np.linalg.inv(lhs),rhs)
        aa=aa+dpar[:n]
        bb=bb+dpar[n:]
    return aa,np.append(1,bb)
     

def rateval(aa,bb,x):
    num=np.polyval(np.flipud(aa),x)
    denom=np.polyval(np.flipud(bb),x)
    return num/denom

