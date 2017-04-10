#With the use of Sparse matrix (sparse matrix is implemented by using dictionary)
from math import *
import numpy as np
from datetime import datetime

def loadMovieLens(path='again_data_divide', file):
    # Load data
    prefs={}
    for line in open(path+file):
        (user,movieid,rating)=line.split(':')
        prefs.setdefault(int(user)-1,{})
        prefs[int(user)-1][int(movieid)-1]=float(rating)
    return prefs

def loadMovieLens2(path='again_data_divide', file1, file2):
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

def matrix_factorization(R,N, M,K=15,steps=15, alpha=0.005, beta=0.1):

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
				eij=R[i][j] - np.dot(P[i,:],Q[:,j])
				# for k in range(K) :
				# 	eij = eij + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
				for k in range(K) :
					P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
					Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
				BU[i]=BU[i]+alpha*(eij-beta*BU[i]);
				BI[j]=BI[j]+alpha*(eij-beta*BI[j]);

				e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
				for k in range(K) :
					e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
				e = e + (beta/2)*(pow(BU[i],2) + pow(BI[j],2))
		if e < 0.001:
			break

	return P, Q.T



if __name__=='__main__':
    
    f=open("crossbiasMFresults.dat","w")
    tmp=10
    while tmp<=100:

        st=datetime.now()
        trainPrefs = loadMovieLens2('/d1.dat','/d'+str(tmp)+'Train.dat')
        testPrefs = loadMovieLens('/d'+str(tmp)+'Test.dat')
        N=6040
        M=3952
        steps=15
        alpha=0.005
        beta=0.1
        K=15
        nP, nQ = matrix_factorization(trainPrefs,N, M)
        final_MAE=[]
        final_RMSD=[]
        true_positive=0
        total=0             
        #calculating error (MAE)
        for i in testPrefs :
                for j in testPrefs[i] :
                    total+=1
                    predcitedRating=np.dot(nP[i,:],nQ.T[:,j])
                    actualRating=testPrefs[i][j]
                    MAE=fabs(predcitedRating-actualRating)
                    final_MAE.append(MAE)
                    RMSD=MAE*MAE
                    final_RMSD.append(RMSD)
                    #liking
                    if predcitedRating>=2.5 and actualRating>=2.5:
                        true_positive+=1
                    #disliking  
                    elif predcitedRating<=2.5 and actualRating<=2.5:
                        true_positive+=1

        FMAE=(sum(final_MAE)/len(final_MAE))
        FRMSE=sqrt(sum(final_RMSD)/len(final_RMSD))
        PRECISION=(1.0*true_positive/total)*100
        RECALL=100

        print ("%d final MAE=%lf, RMSD=%lf %lf %lf" % (tmp,FMAE,FRMSE,PRECISION,RECALL))

        f.write("%d %f %f %f %f\n" %(tmp,FMAE,FRMSE,PRECISION,RECALL))
        tmp+=10
    f.close()    