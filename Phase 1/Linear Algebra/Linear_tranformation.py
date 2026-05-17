import numpy as np

def matrix_vector_multiply_raw(m, v): # m is matrix and v is vector
    if len(m[0]) != len(v):
        raise ValueError('Dimension mismatch. Number of columns in matrix not equals to length of the vector')

    result = []
    
    for i in range(len(m)):
        element_sum = 0
        for j in range(len(v)):
            element_sum += m[i][j]*v[j]
        result.append(element_sum)

    return np.array(result)

m = np.array([[0, -1], [1, 0]])
v = np.array([1, 0])
raw_sol = matrix_vector_multiply_raw(m, v)
print(f'Raw solution: {raw_sol}')
print('The matrix multiplicaiton here is simply the linear transformaiton applied on the given vector v (v is a vector in initial space where basis vectors are i^ and j^) to the new space where column 0 of m and column 1 of m are the respective positions where the initial basis vector landed')
print('For the above specific test case, the final geometric location of the initial vector is [0, 1]. The vector rotated by 90 degrees towards +ve y-axis')
print(f'Numpy version: {np.matmul(m, v)}')