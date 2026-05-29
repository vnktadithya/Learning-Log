import numpy as np

def vector_add_raw(x, y):
    if len(x) != len(y):
        raise ValueError('Dimensionality mismatch. Cannot perform Vector Addition')
    return np.array([x[i] + y[i] for i in range(len(x))])

def scalar_mul_raw(x, y): # x is a scalar
    return np.array([x*y[i] for i in range(len(y))])


x = np.array([1, 0, -3])
y = np.array([2, 34.9, 7.6])
vector_add_res = vector_add_raw(x, y)
print(f'vector addition raw version: {vector_add_res}')
print(f'vector addition using numpy: {np.add(x, y)}')
print('Vector addition requires them to be of same dimension. Adding two vectors adds the elements at same positions in both the vectors and produce the final resultant vector with the same dimension as inputs.')


x1 = 4.876
scalar_mul_res = scalar_mul_raw(x1, y)
print(f'scalar multiplication raw version: {scalar_mul_res}')
print(f'scalar multiplication using numpy: {np.multiply(y, x1)}')
print('Multiplying the vector y with a scalar x scales each value of the vector y by x and the direction of the resultant vector depends on the sign of the scalar value.')
