from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def average_errors(arr):
    max_length = max(len(x) for x in arr)

    # Create a new array to hold the padded values
    padded_arr = np.zeros((len(arr), max_length))

    # Loop over each list in arr and pad it with zeros
    for i, x in enumerate(arr):
        padded_arr[i,:len(x)] = x

    # Calculate the mean array
    mean_array = np.sum(padded_arr, axis=0) / np.count_nonzero(padded_arr, axis=0)
    return mean_array

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
X = np.array(X)
Y = np.array(Y).reshape(-1,1)
errors_train = []
errors_test = []
for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    listx = solve_least_squares(X_train,Y_train)
    err_train = [err(X_train,Y_train,x) for x in listx]
    err_test = [err(X_test,Y_test,x) for x in listx]
    errors_train.append(err_train)
    errors_test.append(err_test)


mean_train = average_errors(errors_train)
mean_test = average_errors(errors_test)

iterations = np.arange(start=0, stop=len(mean_train), step=1)
plt.figure()
plt.plot(iterations, mean_train, label='mean train err')
plt.plot(iterations, mean_test, label='mean test err')

# Add labels and title
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Error of each iteration')
plt.legend()

# Show plot
plt.show()