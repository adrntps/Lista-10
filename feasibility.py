import cvxpy as cp
import numpy as np

Id = np.identity(2)

ket_v0 = np.zeros(shape=(2,1), dtype='complex')
ket_v0[0,0] = 1
trans_ket_v0 = np.transpose(ket_v0)
dagket_v0 = np.conjugate(trans_ket_v0)
sigma0D0 = (1/2)*np.matmul(ket_v0,dagket_v0)

ket_v1 = np.zeros(shape=(2,1), dtype='complex')
ket_v1[1,0] = 1
trans_ket_v1 = np.transpose(ket_v1)
dagket_v1 = np.conjugate(trans_ket_v1)
sigma1D0 = (1/2)*np.matmul(ket_v1,dagket_v1)

ket_v2 = np.zeros(shape=(2,1), dtype='complex')
ket_v2[0,0] = ket_v2[1,0] = 1/(2)**(1/2)
trans_ket_v2 = np.transpose(ket_v2)
dagket_v2 = np.conjugate(trans_ket_v2)
sigma0D1 = (1/2)*np.matmul(ket_v2,dagket_v2)

ket_v3 = np.zeros(shape=(2,1), dtype='complex')
ket_v3[0,0] = 1/(2)**(1/2)
ket_v3[1,0] = -1/(2)**(1/2)
trans_ket_v3 = np.transpose(ket_v3)
dagket_v3 = np.conjugate(trans_ket_v3)
sigma1D1 = (1/2)*np.matmul(ket_v3,dagket_v3)

sigma_lambda1 = cp.Variable(shape=(2,2))
sigma_lambda2 = cp.Variable(shape=(2,2))
sigma_lambda3 = cp.Variable(shape=(2,2))
sigma_lambda4 = cp.Variable(shape=(2,2))
mu = cp.Variable(complex = 'true')

objective = cp.Maximize(cp.real(mu))
constraints = [sigma0D0 == sigma_lambda1 + sigma_lambda2, sigma1D0 == sigma_lambda3 + sigma_lambda4, 
               sigma0D1 == sigma_lambda1 + sigma_lambda3, sigma1D1 == sigma_lambda2 + sigma_lambda4,
               sigma_lambda1 - mu*Id >> 0, sigma_lambda2 - mu*Id >> 0, sigma_lambda3 - mu*Id >> 0,
               sigma_lambda4 - mu*Id >> 0]

prob = cp.Problem(objective, constraints)
result = prob.solve()
print(result)