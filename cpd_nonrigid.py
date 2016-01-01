import numpy as np
from numpy.linalg import inv
import scipy.sparse
from cpd_P import cpd_P

def cpd_nonrigid(X,Y,w,lamb,beta):
    # Construct G:

    G=Y[:,np.newaxis,:]-Y
    G=G*G
    G=np.sum(G,2)
    G=np.exp(-1.0/(2*beta*beta)*G)

    [N,D]=X.shape
    [M,D]=Y.shape
    T=Y
    sigma2=(M*np.trace(np.dot(np.transpose(X),X))+N*np.trace(np.dot(np.transpose(Y),Y))
              +2*np.dot(sum(X),np.transpose(sum(Y))))/(M*N*D)
    sigma2_init=sigma2
    max_it=150
    iter=0
    # eps=np.spacing(1)

    while (iter<max_it) and (sigma2>np.power(10.0,-5)):
        [P1,Pt1,PX]=cpd_P(X,T,sigma2,0.1,M,N,D)
        dp=scipy.sparse.spdiags(P1.T,0,M,M)
        W=np.dot(np.linalg.inv(dp*G+lamb*sigma2*np.eye(M)),(PX-dp*Y)) 
        T=Y+np.dot(G,W)
        Np=sum(P1)
        sigma2=abs((sum(sum(X*X*np.matlib.repmat(Pt1,1,D)))+sum(sum(T*T*np.matlib.repmat(P1,1,D)))-
                  2*np.trace(np.dot(PX.T,T)))/(Np*D))
        iter=iter+1

    return T

    