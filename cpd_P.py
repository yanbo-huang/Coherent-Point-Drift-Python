import numpy as np
from math import pi
def cpd_P(X,Y,sigma2,w,M,N,D):

    G=X[:,np.newaxis,:]-Y
    G=G*G
    G=np.sum(G,2)
    G=np.exp(-1.0/(2*sigma2)*G)
    G1=np.sum(G,1)
    temp2=(G1+(2*pi*sigma2)**(D/2)*w/(1-w)*(M/N)).reshape([N,1])
    P=(G/temp2).T

    P1=(np.sum(P,1)).reshape([M,1])
    PX=np.dot(P,X)
    Pt1=(np.sum(np.transpose(P),1)).reshape([N,1])
    return P1,Pt1,PX

