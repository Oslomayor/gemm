# Author: duzt
# Date: 2023-06-25 16:45
# GEMM 128 tile
# A x B = C

import numpy
import numpy as np

# data
org_A = np.random.randint(-100, 101, size=(128, 128)).astype('float32')
org_B = np.random.randint(-100, 101, size=(128, 128)).astype('float32')
org_C = np.zeros((128, 128), 'float32')

# ref out
ref_C = np.matmul(org_A, org_B)

# GEMM tile
A = org_A.reshape(128, 8, 1, 16).copy()
B = org_B.reshape(128, 128).copy()
C = org_C.reshape(128, 8, 1, 16).copy()

# reshape
tB = np.zeros((8, 8, 16, 16), 'float32')
for i in range(0, 8):
    for j in range(0, 8):
        tB[i, j, :, :] = B[16*i:16+16*i, 16*j:16+16*j]

C = org_C.reshape(128, 8, 1, 16).copy()

# impl gemm mini
def gemm_mini(A:numpy.array, B:numpy.array) -> numpy.array:
    assert A.shape == (1, 16)
    assert B.shape == (16, 16)
    return A @ B

# impl gemm 128
for b in range(0, 128):
    for c in range(0, 8):
        for t in range(0, 8):
            # gemm mini
            # C[b, c, :, :] += A[b, t, :, :] @ tB[t, c, :, :]
            C[b, c, :, :] += gemm_mini(A[b, t, :, :], tB[t, c, :, :])

C = C.reshape(128, 128)
# reshape:
# This will be a new view object if possible; otherwise, it will be a copy.
# The data buffer remains the same, so any changes made to a view reflects in the original copy.
# A copy can be forced by using ndarray.copy()
np.testing.assert_array_equal(ref_C, C)

# GEMM tile init
C2 = org_C.reshape(128, 8, 1, 16).copy()
# 证明外两层 for 没问题
for b in range(0, 128):
    for c in range(0, 8):
        C2[b, c, :, :] = A[b, :, :, :].reshape(1,128) @ tB[:, c, :, :].reshape(128, 16)

C2 = C2.reshape(128, 128)
np.testing.assert_array_equal(ref_C, C2)
