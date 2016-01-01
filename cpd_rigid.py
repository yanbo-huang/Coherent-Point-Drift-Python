import numpy as np
from numpy.linalg import inv
from cpd_P import cpd_P

def cpd_rigid(X,Y,w):
    
    [N,D]=X.shape
    [M,D]=Y.shape
    T=Y
    sigma2=(M*np.trace(np.dot(np.transpose(X),X))+N*np.trace(np.dot(np.transpose(Y),Y))
              +2*np.dot(sum(X),np.transpose(sum(Y))))/(M*N*D)
    sigma2_init=sigma2
    max_it=150
    iter=0
    # eps=np.spacing(1)

    while (iter<max_it) and (sigma2>np.power(10.0,-8)):
    
        [P1,Pt1,PX]=cpd_P(X,T,sigma2,0.0,M,N,D)
        Np=sum(Pt1)
        mu_x=np.dot(np.transpose(X),Pt1)/Np
        mu_y=np.dot(np.transpose(Y),P1)/Np
        A=np.dot(np.transpose(PX),Y)-Np*(np.dot(mu_x,np.transpose(mu_y)))
        [U,S,V] = np.linalg.svd(A)
        S=np.diag(S)
        C=np.eye(D)
        C[-1,-1]=np.linalg.det(np.dot(U,V))
        R=np.dot(U,np.dot(C,V))
        sigma2save=sigma2
        s=np.trace(np.dot(S,C))/(sum(sum(Y*Y*np.matlib.repmat(P1,1,D)))-Np*\
                np.dot(np.transpose(mu_y),mu_y))
        sigma22=abs(sum(sum(X*X*np.matlib.repmat(Pt1,1,D)))-Np*\
                np.dot(np.transpose(mu_x),mu_x)-s*np.trace(np.dot(S,C)))/(Np*D)
        sigma2=sigma22[0][0]
        t=mu_x-np.dot(s*R,mu_y)
        T=np.dot(s*Y,np.transpose(R))+np.matlib.repmat(np.transpose(t),M,1)
        iter=iter+1
    return T