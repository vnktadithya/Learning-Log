import numpy as np

def matrix_multiply_raw(m1, m2):

    if (len(m1[0]) != len(m2)):
        raise ValueError('Dimension mismatch. Cannot perform Matrix Multiplication')

    result = []

    for i in range(len(m1)):
        row = []
        for j in range(len(m2[0])):
            element_sum = 0
            for k in range(len(m2)):
                element_sum += m1[i][k]*m2[k][j]
            row.append(element_sum)
        result.append(row)  

    return np.array(result)

m1 = np.array([[1, 1, -1], [0, 1, 0]])
m2 = np.array([[0, -1], [1, 0], [1, -1]])

m1m2_result = matrix_multiply_raw(m1, m2)
m2m1_result = matrix_multiply_raw(m2, m1)
print(f"m1*m2 raw result: {m1m2_result}")
print(f"m2*m1 raw result: {m2m1_result}")
print(f"m1*m2 numpy resut: {np.matmul(m1, m2)}")
print(f"m2*m1 numpy resut: {np.matmul(m2, m1)}")

print("In this case, by performing the shear transformaiton first, i^ stays the same but j^ tilts towards right and the space is streched. Now applying the rotaion by 90 degrees, the initial i^ basis vector transforms to be in +ve y-axis direction and the initial j^ now lies at the position [-1, 1].")
print("If we perform rotation first, initial i^ now sits on +ve y-axis and initial j^ now sits on -ve x-axis. After this, if we apply shear transformation, then we get a stretch in the space in a different axis between +ve y-axis and -ve x-axis.")