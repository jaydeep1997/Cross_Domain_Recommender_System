#With the use of Sparse matrix (sparse matrix is implemented by using dictionary)
from math import *
import numpy as np
from datetime import datetime



def loadMovieLens(file,path='again_data_divide'):
    # Load data
    prefs={}
    for line in open(path+file):
        (user,movieid,rating)=line.split(':')
        prefs.setdefault(int(user)-1,{})
        prefs[int(user)-1][int(movieid)-1]=float(rating)
    return prefs

def matrix_factorization(R,N, M,K=15,steps=15, alpha=0.005, beta=0.05):
	P = np.random.rand(N,K)
	Q = np.random.rand(M,K)
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



if __name__=='__main__':

    f=open("singleMF_result","w")
    tmp=10
    while tmp<=100:
        trainPrefs = loadMovieLens('/d'+str(tmp)+'Train.dat')
        testPrefs = loadMovieLens('/d'+str(tmp)+'Test.dat')	
        N=6040
        M=3952
        steps=15
        alpha=0.005
        beta=0.05
        k=15
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
