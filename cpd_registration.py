from cpd_affine import cpd_affine
from cpd_rigid import cpd_rigid
from cpd_nonrigid import cpd_nonrigid

def cpd_registration(method,X,Y,w,lamb=3.0,beta=2.0):

    if method=='affine':
        T=cpd_affine(X,Y,w)
    elif method=='rigid':
        T=cpd_rigid(X,Y,w)
    elif method=='nonrigid':
        T=cpd_nonrigid(X,Y,w,lamb,beta)
    else:
        print "Please input a valid point set registration method"
    
    return T