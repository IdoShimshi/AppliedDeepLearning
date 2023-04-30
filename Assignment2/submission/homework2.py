from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def get_dataset():
    X, Y = datasets.load_diabetes(return_X_y=True)
    X = np.array(X)
    Y = np.array([Y]).T
    return X, Y


def solve_least_squares(X, y, delta=50, eps=0.01):
    steps = []
    xk = np.random.rand(X.shape[1], 1)
    while (np.linalg.norm(least_squares_gradient(X, y, xk))) > delta:
        steps.append(xk)
        xk = xk - eps * least_squares_gradient(X, y, xk)
    steps.append(xk)
    return steps


def least_squares_gradient(A, b, x):
    ATA = A.T @ A
    ATb = A.T @ b
    return (ATA @ x) - ATb


def get_error(X, Y, x):
    return (np.linalg.norm((X @ x) - Y) ** 2) / X.shape[0]


def task1():
    X, Y = get_dataset()
    least_squares_steps = solve_least_squares(X, Y)
    print(f"Final error: {get_error(X, Y, least_squares_steps[-1])}")

    train_errors = [get_error(X, Y, xk) for xk in least_squares_steps]
    plt.plot(train_errors)
    plt.xlabel('Step Index')
    plt.ylabel('Error')
    plt.title('Error of each step')
    plt.show()


def task2():
    X, Y = get_dataset()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    least_squares_steps = solve_least_squares(X_train, Y_train)
    print(f"Final train error: {get_error(X_train, Y_train, least_squares_steps[-1])}")
    print(f"Final test error: {get_error(X_test, Y_test, least_squares_steps[-1])}")

    train_errors = [get_error(X_train, Y_train, xk) for xk in least_squares_steps]
    test_errors = [get_error(X_test, Y_test, xk) for xk in least_squares_steps]
    plt.plot(train_errors, label='Train error')
    plt.plot(test_errors, label='Test error')
    plt.xlabel('Step Index')
    plt.ylabel('Error')
    plt.title('Error of each step')
    plt.legend()
    plt.show()


def task3():
    X, Y = get_dataset()

    train_error = np.empty(10)
    test_error = np.empty(10)
    for i in range(0, 10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        least_squares_steps = solve_least_squares(X_train, Y_train)
        train_error[i] = get_error(X_train, Y_train, least_squares_steps[-1])
        test_error[i] = get_error(X_test, Y_test, least_squares_steps[-1])

        train_errors = [get_error(X_train, Y_train, xk) for xk in least_squares_steps]
        test_errors = [get_error(X_test, Y_test, xk) for xk in least_squares_steps]
        plt.plot(train_errors, label='Train error')
        plt.plot(test_errors, label='Test error')
        plt.xlabel('Step Index')
        plt.ylabel('Error')
        plt.title('Iteration ' + str(i))
        plt.legend()
        plt.show()

    print(f"min train error: {np.min(train_error)}")
    print(f"min test error: {np.min(test_error)}")
    print(f"mean train error: {np.mean(train_error)}")
    print(f"mean test error: {np.mean(test_error)}")

    plt.plot(train_error, label='Train error')
    plt.plot(test_error, label='Test error')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Final error in every iteration')
    plt.xticks(np.arange(0, 10, 1))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    task1()
    task2()
    task3()
