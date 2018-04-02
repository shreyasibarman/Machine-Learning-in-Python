import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
from scipy.special import expit as sigmoid

data = loadmat('E:\MLshiz\ex3data1.mat')

X=data['X']
y=data['y']
m=X.shape[0]
n=X.shape[1]

Y=np.zeros((m,10))
max=np.amax(y)

hidden_size = 25

theta1 = 0.25*(np.random.random((n + 1, hidden_size))-0.5)
theta2 = 0.25*(np.random.random((hidden_size + 1, Y.shape[1]))-0.5)

for i in range(max):
    var=y[i]
    Y[i,var-1]=1


def sigmoid_grad(z):
    
    s=sigmoid(z)
    return s*(1-s)

def forward_propagate(X,theta1,theta2):
    a1=np.c_[np.ones(X.shape[0]),X]
    z1=a1.dot(theta1)
    a2=np.c_[np.ones(X.shape[0]),sigmoid(z1)]
    z3=a2.dot(theta2)
    h3=sigmoid(z3)

    return a1,z1,a2,z3,h3

def cost(X,Y,t1,t2):
    a1,z1,a2,z3,hyp=forward_propagate(X,t1,t2)
    one=np.ones(Y.shape)
    J=(-1/m)*np.sum(np.multiply(Y,np.log(hyp))+np.multiply((one-Y),np.log(one-hyp)))
    return J

for i in range(100000):
    a1, z1, a2, z3, hyp = forward_propagate(X, theta1, theta2)
    del_2= Y-hyp
    delta2=del_2
    del_1=del_2.dot(theta2[1:,:].T)
    delta1=del_1*sigmoid_grad(z1)

    theta2+=(-0.01)*a2.T.dot(delta2)
    theta1+=(-0.01)*a1.T.dot(delta1)

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

