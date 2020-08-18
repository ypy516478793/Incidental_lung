import numpy as np

A = np.arange(24).reshape([2,3,4])
m = np.random.permutation(np.arange(4))
M = np.zeros([4, 4])
for i in range(len(m)):
    M[m[i], i] = 1
print(M)

print("A before: ")
for i in range(len(m)):
    print(A[..., i])

A_p = np.matmul(A, M)
print("A after permutation: ")
for i in range(len(m)):
    print(A_p[..., i])