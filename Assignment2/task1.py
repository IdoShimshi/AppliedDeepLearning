from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def err(X,Y,x):
    return (np.linalg.norm((X @ x) - Y)**2) / X.shape[0]

def solve_least_squares(X,Y,delta=10,eps=0.01):
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
# X = np.array(X)
# Y = np.array(Y).reshape(-1,1)
# listx = solve_least_squares(X,Y)
# print(err(X,Y,listx[-1]))
# errors = [err(X,Y,x) for x in listx]
# plt.plot(errors)

# # Add labels and title
# plt.xlabel('Step Index')
# plt.ylabel('Error')
# plt.title('Error of each step')

# # Show plot
# plt.show()