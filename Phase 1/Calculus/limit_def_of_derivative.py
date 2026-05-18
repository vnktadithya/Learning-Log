import math

def calculate_numerical_derivative(f, x, h=0.0001):
    return (f(x+h) - f(x))/h

def x_square(a):
    return a**2

def sin_x(a):
    return math.sin(a)

def e_power_x(a):
    return math.exp(a)

def x_cube(a):
    return a**3


print("Evaluating numerical derivates using raw limit definition of derivative:")
print('x^2: ', calculate_numerical_derivative(x_square, 2))
print('sin(x): ', calculate_numerical_derivative(sin_x, 2))
print('e^x: ', calculate_numerical_derivative(e_power_x, 2))

h = [0.1, 0.01, 0.0001, 1e-10, 1e-15, 1e-17]

for i in h:
    print(f'Evaluating the numerical derivative of x^3 at h = {i}', calculate_numerical_derivative(x_cube, 2, i))
