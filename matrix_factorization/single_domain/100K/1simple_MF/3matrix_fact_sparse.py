#http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
#With the use of Sparse matrix (sparse matrix is implemented by using dictionary)
import numpy as np
def matrix_factorization(R, K, N, M, steps=5000, alpha=0.0002, beta=0.02):

	P = np.random.rand(N,K)
	Q = np.random.rand(M,K)

	# print(P)
	# print(Q)

	Q = Q.T

	for step in range(steps):
		e = 0
		for i in R :
			for j in R[i] :
				eij=R[i][j] - np.dot(P[i,:],Q[:,j])
				for k in range(K) :
					P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
					Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

				e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
				for k in range(K) :
					e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
		if e < 0.001:
			break


	return P, Q.T




R={0:{0:5,1:3,3:1},
	1:{0:4,3:1},
	2:{0:1,1:1,3:5},
	3:{0:1,3:4},
	4:{1:1,2:5,3:4}}

# R = [
#      [5,3,0,1],
#      [4,0,0,1],
#      [1,1,0,5],
#      [1,0,0,4],
#      [0,1,5,4],
#     ]

print(R)
N = 5
M = 4
K = 2

print(N,M)


nP, nQ = matrix_factorization(R, K, N, M)
print(nP)
print(nQ)

print(np.dot(nP, nQ.T))								#don't do this because it will take very large time (if n & m are large)
