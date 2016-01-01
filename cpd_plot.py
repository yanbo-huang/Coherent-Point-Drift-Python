import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cpd_plot(X,Y,T):
    if len(X[0,:])==2:
        plt.figure(1)
        plt.plot(X[:,0],X[:,1],'go')
        plt.plot(Y[:,0],Y[:,1],'r+')
        plt.title("Before registration")
        plt.figure(2)
        plt.plot(X[:,0],X[:,1],'go')
        plt.plot(T[:,0],T[:,1],'r+')
        plt.title("After registration")
        plt.show()
    elif len(X[0,:])==3:
        ax1=Axes3D(plt.figure(1))
        ax1.plot(X[:,0],X[:,1],X[:,2],'yo')
        ax1.plot(Y[:,0],Y[:,1],Y[:,2],'r+')
        ax1.set_title("Before registration", fontdict=None, loc='center')
        ax2=Axes3D(plt.figure(2))
        ax2.plot(X[:,0],X[:,1],X[:,2],'yo')
        ax2.plot(T[:,0],T[:,1],T[:,2],'r+')
        ax2.set_title("After registration", fontdict=None, loc='center')
        plt.show()