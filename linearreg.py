import numpy as np
import matplotlib.pyplot as plt


data=np.loadtxt('E:\MLshiz\ex1data1.txt', delimiter=',')
x=data[:,0]
y=np.c_[data[:,1]]
print(x)
X=np.c_[np.ones(data.shape[0]),x]
print(X)
m=X.shape[0]
n=X.shape[1]

iter=1500
alpha=0.01

theta=np.zeros((n,y.shape[1]))


def visualize_data(x,y):
    plt.plot(x,y,'bx')
    plt.show()
visualize_data(x,y)

def get_cost(X,y,theta):
    hyp=X.dot(theta)
    err=hyp-y
    sqr_err=err**2
    J=np.sum(sqr_err)/(2*m)
    return J  

def gradient_descent(X,y,theta):
    cost=[]
    for i in range(0,iter):
        hyp=X.dot(theta)
        theta=theta-((alpha/m)*(X.T.dot(hyp-y)))
        J=get_cost(X,y,theta)
        cost.append(J)
    return theta,cost

def predict(weights,input):
    X= np.c_[np.ones(1),input]
    return (X.dot(weights))

print("Visualize data...")
visualize_data(x,y)

print("The initial cost is:%f" %get_cost(X,y,theta))

print("\nTheta obtained after gradient descent")
weights=gradient_descent(X,y,theta)[0]
print(weights)


print("Plotting line of best fit")
plt.plot(x,y,'bx')
weights=y=np.c_[weights]
eqn=X.dot(weights)
plt.plot(x,eqn,'-')
plt.show()


cost=gradient_descent(X,y,theta)[1]
plt.plot(cost,"-")
plt.ylabel('cost function')
plt.show()

print("For a population of 35000, we predict a profit of %f" %(predict(weights,3.5)*10000))





data=np.loadtxt('E:\MLshiz\ex1data2.txt', delimiter=',')
x=data[:,0:2]
y=np.c_[data[:,1]]

m=np.mean(x,axis=0)
print("Before normalization, x is:")
print(x)

std=np.std(x,axis=0)
for i in range(x.shape[0]):
    x[i,:]=(x[i,:]-m)/std;
print("After normalization, x is:")
print(x)











