import numpy as np
def determinant_2x2_raw(m):
    if len(m) != 2 or len(m[0]) != 2:
        raise ValueError('Dimensions not equals to 2')

    return m[0][0]*m[1][1] - m[0][1]*m[1][0]

m = np.array([[1,2], [2,4]])
det = determinant_2x2_raw(m)
print(f'Determinat: {det}')
print('The determinant value represents that value by which the area is scaled by in the transformed space.')
print('In the above case, the area becomes 0 because the basis vectors in the  transformed space are linearly dependent on each other.')