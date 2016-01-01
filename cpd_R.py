import numpy as np
from math import sin,cos

def cpd_R(n):
    
    if n==3:
        R1=np.eye(3)
        R2=np.eye(3)
        R3=np.eye(3)
        R1[0:2:,0:2]=rot(np.random.rand(1))
        R2[::2,::2]=rot(np.random.rand(1))
        R3[1:,1:]=rot(np.random.rand(1))
        R=np.dot(np.dot(R1,R2),R3)
        
    elif n==2:
        R=rot(np.random.rand(1))
    return R

def rot(f):
    return np.array([[cos(f),-sin(f)],[sin(f),cos(f)]])

def cpd_B(n):
    B=np.eye(n,dtype=int)+0.5*np.absolute(np.random.randn(n,n))
    return B
