from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def err(X,Y,x):
    return (np.linalg.norm((X @ x) - Y)**2) / X.shape[0]

def solve_least_squares(X,Y,delta=0.1,eps=0.5):
    retx = []
    XTY = X.T @ Y
    XTX = X.T @ X
    derivative = lambda x: ((XTX @ x) - (XTY))
    xk = np.random.rand(X.shape[1],1)
    while (np.linalg.norm(derivative(xk))) > delta:
        retx.append(xk)
        xk = xk - eps*derivative(xk)
    retx.append(xk)
    return retx

X,Y = datasets.load_diabetes(return_X_y=True)
X = np.array(X)
Y = np.array([Y]).T
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


listx = solve_least_squares(X_train,Y_train)
errors_train = [err(X_train,Y_train,x) for x in listx]
errors_test = [err(X_test,Y_test,x) for x in listx]
x_index = np.arange(start=0, stop=len(listx), step=1)
plt.figure()
plt.plot(x_index, errors_train, label='train err')
plt.plot(x_index, errors_test, label='test err')

# Add labels and title
plt.xlabel('Step Index')
plt.ylabel('Error')
plt.title('Error of each step')
plt.legend()

# Show plot
plt.show()