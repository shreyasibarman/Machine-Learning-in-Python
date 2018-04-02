import numpy as np
import matplotlib as plt
import scipy.optimize as opt

data=np.loadtxt('E:\MLshiz\ex2data1.txt', delimiter=',')

x= data[:,0:2]
y= np.c_[data[:,2]]

X=np.c_[np.ones((x.shape[0],1)),x]

m=X.shape[0]
n=X.shape[1]

iter=1500
alpha=0.01
l=1

theta=np.zeros((n,y.shape[1]))
print(X.shape)
print(y.shape)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def get_cost(X,y,theta):
    hyp= sigmoid(X.dot(theta))
    one= np.ones(y.shape)
    J=(-1/m)*np.sum(np.multiply(y,np.log(hyp))+np.multiply((one-y),np.log(1-hyp)))
    return J

g=get_cost(X,y,theta)
print("Initial cost is:",g)

def gradient(X,y,theta):
    for i in range(iter):
        hyp=sigmoid(X.dot(theta))
        theta=theta-(alpha/m)*(X.T.dot(hyp-y))
    return theta

g=gradient(X,y,theta)

print("Weights obtained from gradient descent are:")
print(g)

X=np.array([1,45,85])
hyp=sigmoid(X.dot(theta))
print("The predicted value is:",float(hyp))




data=np.loadtxt("E:\MLshiz\ex2data2.txt",delimiter=",")

x=data[:,0:2]
y=np.c_[data[:,2]]

X=np.c_[np.ones((x.shape[0],1)),x]

m=X.shape[0]
n=X.shape[1]

iter=1500
alpha=0.01
l=1

theta=np.zeros((n,y.shape[1]))

theta=np.zeros((n,y.shape[1]))
def regularized_cost():
   J=get_cost(X,y,theta)
   J=J+(1/2*m)*np.sum(theta[:,1:]**2)
   return J

j=regularized_cost()
print("Initial cost after regularization is:",j)

def regularized_descent(X,y,theta):
    for i in range(iter):
        hyp=sigmoid(X.dot(theta))
        var=theta[0,0]
        var2=X.T.dot(hyp-y)[0,0]
        theta[0,0]=var-(alpha/m)*(var2)
        theta[1:,0]=theta[1:,0]-(alpha/m)*((X.T.dot(hyp-y))[1:,0])-(1/m)*theta[1:,0]
    return theta

rd=regularized_descent(X,y,theta)
print("Theta values obtained after regularized gradient descent are")
print(rd)

X=np.array([1,-0.25,1.5])
hyp=sigmoid(X.dot(rd))
print("The predicted value is:",float(hyp))






