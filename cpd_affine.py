import numpy as np
from numpy.linalg import inv
from math import *
from cpd_P import cpd_P

def cpd_affine(X,Y,w):
    [N,D]=X.shape
    [M,D]=Y.shape
    T=Y
    sigma2=(M*np.trace(np.dot(np.transpose(X),X))+N*np.trace(np.dot(np.transpose(Y),Y))
              +2*np.dot(sum(X),np.transpose(sum(Y))))/(M*N*D)
    sigma2_init=sigma2
    max_it=150
    iter=0
    eps=np.spacing(1)

    while (iter<max_it) and (sigma2>10*eps):
    
        [P1,Pt1,PX]=cpd_P(X,T,sigma2,0.0,M,N,D)
        #precompute
        Np=sum(P1)
        mu_x=np.dot(np.transpose(X),Pt1)/Np
        mu_y=np.dot(np.transpose(Y),P1)/Np
        #solve for parameters
        B1=np.dot(np.transpose(PX),Y)-Np*(np.dot(mu_x,np.transpose(mu_y)))
        B2=np.dot(np.transpose(Y*np.matlib.repmat(P1,1,D)),Y)-Np*np.dot(mu_y,np.transpose(mu_y))
        B=np.dot(B1,inv(B2))
        t=mu_x-np.dot(B,mu_y)
        sigma2save=sigma2
        sigma22=abs(sum(sum(X*X*np.matlib.repmat(Pt1,1,D)))-Np*\
                np.dot(np.transpose(mu_x),mu_x)-np.trace(np.dot(B1,np.transpose(B))))/(Np*D)
        sigma2=sigma22[0][0]
    
        T=np.dot(Y,np.transpose(B))+np.matlib.repmat(np.transpose(t),M,1)
        iter=iter+1
    
    return T