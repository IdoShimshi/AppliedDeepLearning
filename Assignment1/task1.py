from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def err(X,Y,x):
    return np.linalg.norm((X @ x) - Y)

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
X = np.array(X)
Y = np.array([Y]).T
listx = solve_least_squares(X,Y)
print(err(X,Y,listx[-1]))
errors = [err(X,Y,x) for x in listx]
plt.plot(errors)

# Add labels and title
plt.xlabel('Vector Index')
plt.ylabel('Error')
plt.title('Error of Each Vector')

# Show plot
plt.show()