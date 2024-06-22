import cvxpy as cp
import numpy as np

Id = np.identity(2)

# Definindo o Assemblage. O estado quântico é um maximamente emaranhado de dois qubits; em particular, 
#o estado singleto (por simplicidade). As medições de Alice são A_0 = \sigma_z e A_1 = \sigma_x. Com tais
#informações, determinar os estados não normalizados \sigma_{a|x} podem ser prontamente determinados. Sobre
#a notação adotada, o índice 0 refere-se à medição \sigma_z e o índice 1 à medição \sigma_x; concernentes às
#respostas, 0 indica a resposta "+1" e 1 indica a resposta "-1". 

ket_v0 = np.zeros(shape=(2,1), dtype='complex')
ket_v0[0,0] = 1
trans_ket_v0 = np.transpose(ket_v0)
dagket_v0 = np.conjugate(trans_ket_v0)
sigma1D0 = (1/2)*np.matmul(ket_v0,dagket_v0)

ket_v1 = np.zeros(shape=(2,1), dtype='complex')
ket_v1[1,0] = 1
trans_ket_v1 = np.transpose(ket_v1)
dagket_v1 = np.conjugate(trans_ket_v1)
sigma0D0 = (1/2)*np.matmul(ket_v1,dagket_v1)

ket_v2 = np.zeros(shape=(2,1), dtype='complex')
ket_v2[0,0] = ket_v2[1,0] = 1/(2)**(1/2)
trans_ket_v2 = np.transpose(ket_v2)
dagket_v2 = np.conjugate(trans_ket_v2)
sigma1D1 = (1/2)*np.matmul(ket_v2,dagket_v2)

ket_v3 = np.zeros(shape=(2,1), dtype='complex')
ket_v3[0,0] = 1/(2)**(1/2)
ket_v3[1,0] = -1/(2)**(1/2)
trans_ket_v3 = np.transpose(ket_v3)
dagket_v3 = np.conjugate(trans_ket_v3)
sigma0D1 = (1/2)*np.matmul(ket_v3,dagket_v3)

#Definindo as variáveis de otimização do problema: F_{a|x}. 

F0D0 = cp.Variable(shape=(2,2), hermitian = True)
F1D0 = cp.Variable(shape=(2,2), hermitian = True)
F0D1 = cp.Variable(shape=(2,2), hermitian = True)
F1D1 = cp.Variable(shape=(2,2), hermitian = True)

# Aplicação direta da SDP proposta no artigo de review sobre Quantum Steering (Daniel Cavalcanti).
G = cp.matmul(F0D0,sigma0D0) + cp.matmul(F0D1,sigma0D1) + cp.matmul(F1D0,sigma1D0) + cp.matmul(F1D1,sigma1D1)
objective = cp.Minimize(cp.real(cp.trace(G)))
constraints = [F0D0 + F0D1 >> 0, F0D0 + F1D1 >> 0, F1D0 + F0D1 >> 0, F1D0 + F1D1 >> 0,
               cp.trace(F0D0 + F0D1 + F1D0 + F1D1) == 1/2]
prob = cp.Problem(objective, constraints)
result = prob.solve()
print(result)
print()
print('F0D0: ', F0D0.value)
print()
print('F1D0: ',F1D0.value)
print()
print('F0D1: ',F0D1.value)
print()
print('F1D1: ',F1D1.value)