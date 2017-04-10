#With the use of Sparse matrix (sparse matrix is implemented by using dictionary)
from math import *
import numpy as np
from datetime import datetime

def loadMovieLens(path='/home/jaydeep/Desktop/Mini_project/movielens_dataset/movielens_1m/chaudhary_devide', file='/test.dat'):
    # Load data
    prefs={}
    for line in open(path+file):
        (user,movieid,rating)=line.split(':')
        prefs.setdefault(int(user)-1,{})
        prefs[int(user)-1][int(movieid)-1]=float(rating)
    return prefs

def loadMovieLens2(path='/home/jaydeep/Desktop/Mini_project/movielens_dataset/movielens_1m/chaudhary_devide', file1='/train1.dat', file2='/train2.dat'):
    # Load data
    prefs={}
    for line in open(path+file1):
        (user,movieid,rating,ts)=line.split('::')
        prefs.setdefault(int(user)-1,{})
        prefs[int(user)-1][int(movieid)-1]=float(rating)
    for line in open(path+file2):
        (user,movieid,rating)=line.split(':')
        prefs.setdefault(int(user)-1,{})
        prefs[int(user)-1][int(movieid)-1]=float(rating)

    return prefs

def matrix_factorization(R, K, N, M, steps=5000, alpha=0.0002, beta=0.02):

	P = np.random.rand(N,K)
	Q = np.random.rand(M,K)
	BU = np.random.rand(N,1)
	BI = np.random.rand(M,1)

	# print(P)
	# print(Q)

	Q = Q.T

	for step in range(steps):
		e = 0
		for i in R :
			for j in R[i] :
				eij=R[i][j] - np.dot(P[i,:],Q[:,j])									#eij=R[i][j] - mu - BU[i] - BI[j] - np.dot(P[i,:],Q[:,j])
				# for k in range(K) :
				# 	eij = eij + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
				for k in range(K) :
					P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
					Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
				BU[i]=BU[i]+alpha*(eij-beta*BU[i]);									#BU[i]=BU[i]+alpha*(2*eij-beta*BU[i]);
				BI[j]=BI[j]+alpha*(eij-beta*BI[j]);

				e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)						#e = e + pow(R[i][j] - mu - BU[i] - BI[j] - np.dot(P[i,:],Q[:,j]), 2)
				for k in range(K) :
					e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
				e = e + (beta/2)*(pow(BU[i],2) + pow(BI[j],2))
		if e < 0.001:
			break

	return P, Q.T																	#return P, Q.T, BU, BI



if __name__=='__main__':
    st=datetime.now()
    trainPrefs = loadMovieLens2(file1='/d1.dat',file2='/d60Train.dat')
    testPrefs = loadMovieLens(file='/d60Test.dat')

    f=open("test60.dat","w")
    
    #print("load_time=", datetime.now()-st)
    #print("makeR_time=", datetime.now()-st)

    #print(trainPrefs[384][496])

    N=6040
    M=3952
    steps=10

    Ks=[15,20]
    alphas=[0.001,0.005]
    betas=[0.05,0.1]


    for K in Ks : 
        for alpha in alphas :
            for beta in betas :

                nP, nQ = matrix_factorization(trainPrefs, K, N, M, steps, alpha, beta)

                print("MF_time=", datetime.now()-st)
                
                # print(nP)
                # print(nQ)

                total_err=[]
                #calculating error (MAE)
                for i in testPrefs :
                        for j in testPrefs[i] :
                            diff=fabs(np.dot(nP[i,:],nQ.T[:,j])-testPrefs[i][j])					#diff=fabs(mu+BU[i]+BI[ij]+np.dot(nP[i,:],nQ.T[:,j])-testPrefs[i][j])
                            total_err.append(diff)

                print(K,steps)
                MAE=(sum(total_err)/len(total_err))
                print("MAE=%lf" % MAE)
                print("time=", datetime.now()-st)

                f.write("%f %f %f %f\n" %(MAE,alpha,beta,K))
