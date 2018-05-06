from numpy import *
import numpy as np
import matplotlib.pyplot as plt

# final
def sum_of_squares_error(coeff, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - equation(coeff, x)) ** 2
    return totalError / float(len(points))

# final
def step_gradient(current_coeff, points, learningRate):
    gradient = np.zeros(len(current_coeff))
    
    N = float(len(points))
    for p in range(0, len(points)):
        x = points[p, 0]
        y = points[p, 1]
        result = equation(current_coeff, x)
        for i in range(len(current_coeff)):
            gradient[i] += -(2/N) * derivative(i, current_coeff, x) * (y - result)
            
    new_coeff = current_coeff - (learningRate * gradient)
    return new_coeff

def gradient_descent_runner(points, starting_coeff, xmin, xmax, learning_rate, num_iterations):
    coeff = starting_coeff[:]
    for i in range(num_iterations):
        coeff = step_gradient(coeff, array(points), learning_rate)
        
        x = np.linspace(xmin, xmax, 10)
        vfunc = np.vectorize(equation, otypes=[float], excluded=['coeff'])
        y = vfunc(coeff=coeff, x=x)
        
        plt.plot(x, y,'g')
        print("After {0} iterations coeff = {1}, error = {2}".format(i+1, coeff, sum_of_squares_error(coeff, points)))
    return coeff


# polynomial equation
def equation(coeff, x):
    #return coeff[0] * x**2 + coeff[1] * x**1 + coeff[2] * x**0
    polynomial = 0
    for i in range(rank):
        polynomial += coeff[rank - 1 - i] * x**i
    return polynomial

def derivative(index, coeff, x):
    return x**(rank-index)


rank=4
points = genfromtxt("data.csv", delimiter=",")
#print(points[:,[0]])
#print(points[:,[1]])
xmin = min(points[:,[0]])
xmax = max(points[:,[0]])
plt.plot(points[:,[0]], points[:,[1]],'bo')



coeff = np.zeros(rank)
print("After {0} iterations coeff = {1}, error = {2}".format(0, coeff, sum_of_squares_error(coeff, points)))

learning_rate = 0.0000000000001
num_iterations = 100
coeff = gradient_descent_runner(points, coeff, xmin, xmax, learning_rate, num_iterations)

x = np.linspace(xmin, xmax, 10)
vfunc = np.vectorize(equation, otypes=[float], excluded=['coeff'])
y = vfunc(coeff=coeff, x=x)
plt.plot(x, y,'r')

plt.show()