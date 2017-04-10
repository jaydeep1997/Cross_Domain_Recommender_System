#With the use of matrix of size 943X1682, without using sparse matrix
from math import *
import numpy as np
from datetime import datetime


def loadMovieLens(path='/home/jaydeep/Desktop/Mini_project/movielens_dataset/movielens_100K', file='/u1.base'):
	# Get movie titles
	movies={}
	for line in open(path+"/u.item",encoding='latin-1'):
		(id,title)=line.split('|')[0:2]
		movies[id]=title

	# Load data
	prefs={}
	for line in open(path+file):
		(user,movieid,rating,ts)=line.split('\t')
		prefs.setdefault(user,{})
		prefs[user][movieid]=float(rating)
	return prefs


def matrix_factorization(R, P, Q, K, steps=50, alpha=0.0002, beta=0.02):
	Q = Q.T
	for step in range(steps):
		for i in range(len(R)):
		    for j in range(len(R[i])):
		        if R[i][j] > 0:
		            eij = R[i][j] - np.dot(P[i,:],Q[:,j])
		            for k in range(K):
		            	#P[i][k] = 0
		            	#Q[k][j] = 0
		                P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
		                Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
		eR = np.dot(P,Q)
		e = 0
		for i in range(len(R)):
		    for j in range(len(R[i])):
		        if R[i][j] > 0:
		            e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)                      #or e = e + pow(R[i][j] - eR[i][j], 2)
		            for k in range(K):
		                e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
		if e < 0.001:
		    break
	return P, Q.T



if __name__=='__main__':
	st=datetime.now()
	trainPrefs = loadMovieLens(file="/u1.base")
	testPrefs = loadMovieLens(file='/u1.test')

	n_users=943
	n_movies=1682
	
	#print("load_time=", datetime.now()-st)
	

	# if '100' not in trainPrefs['1']:
	# 	print("j")
	
	#R[i][j]: rating given by (i+1)th userId to (j+1)th movieId
	R=[[0 for j in range(n_movies)] for i in range(n_users)]
	for	i in range(n_users):
		for j in range(n_movies):
			if str(i+1) in trainPrefs :
				if str(j+1) not in trainPrefs[str(i+1)] :
					R[i][j]=0
				else :
					R[i][j]=trainPrefs[str(i+1)][str(j+1)]
	
	#print("makeR_time=", datetime.now()-st)
	
	R = np.array(R)
	print(R)
	N = len(R)
	M = len(R[0])
	K = 10
	steps=5

	P = np.random.rand(N,K)
	Q = np.random.rand(M,K)

	#print(P)
	#print(Q)
	
	print("makePQ_time=", datetime.now()-st)
	
	nP, nQ = matrix_factorization(R, P, Q, K, steps)
	
	print("MF_time=", datetime.now()-st)
	
	nR = np.dot(nP, nQ.T)
	#print(nP)
	#print(nQ)
	print(nR)
	
	total_err=[]
	#calculating error(MAE)
	for	i in range(n_users):
		for j in range(n_movies):
			if str(i+1) in testPrefs :
				if str(j+1) in testPrefs[str(i+1)] :
					diff=fabs(nR[i][j]-testPrefs[str(i+1)][str(j+1)])
					total_err.append(diff)
					#print(diff)
	
	print(K,steps)
	print("MAE=%lf" % (sum(total_err)/len(total_err)))
	print("time=", datetime.now()-st)

#MAE=1.343107, time=0:01:09.931801 (K=2, steps=10)
#MAE=0.954393, time=0:01:11.646182 (K=10, steps=5)
#MAE=0.988, 1.002 (K=2, steps=20)
#MAE=0.876567,  (K=5, steps=20)
#MAE=0.789145 (K=5, steps=50)

