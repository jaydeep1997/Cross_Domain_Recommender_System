#With the use of Sparse matrix (sparse matrix is implemented by using dictionary)
from math import *
import numpy as np
from datetime import datetime

def loadMovieLens(path='.', file='/Train.dat'):

    # Load data
    prefs={}
    for line in open(path+file):
        (user,movieid,rating)=line.split(':')
        prefs.setdefault(int(user)-1,{})
        prefs[int(user)-1][int(movieid)-1]=float(rating)
    return prefs


def matrix_factorization(R, K, N, M, steps=50, alpha=0.0002, beta=0.02):

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    
    print("makePQ_time=", datetime.now()-st)

    # print(P)
    # print(Q)

    Q = Q.T

    for step in range(steps):
        e = 0
        for i in R :
            for j in R[i] :
                eij=R[i][j] - np.dot(P[i,:],Q[:,j])
                for k in range(K) :
                    #P[i][k] = 5+5+5
                    #Q[k][j] = 5+5+5
                    P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                    Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

                e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                for k in range(K) :
                    e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break


    return P, Q.T



if __name__=='__main__':
    st=datetime.now()
    trainPrefs = loadMovieLens(file="/Train.dat")
    testPrefs = loadMovieLens(file='/Test.dat')
    
    #print("load_time=", datetime.now()-st)
    #print("makeR_time=", datetime.now()-st)

    #print(trainPrefs[384][496])

    N=6040
    M=3952
    K=1
    steps=1

    nP, nQ = matrix_factorization(trainPrefs, K, N, M, steps, alpha=0.001, beta=0.01)
    
    print("MF_time=", datetime.now()-st)
    
    # print(nP)
    # print(nQ)
    
    total_err=[]
    #calculating error (MAE)
    for i in testPrefs :
            for j in testPrefs[i] :
                diff=fabs(np.dot(nP[i,:],nQ.T[:,j])-testPrefs[i][j])
                total_err.append(diff)
    
    print(K,steps)
    print("MAE=%lf" % (sum(total_err)/len(total_err)))
    print("time=", datetime.now()-st)

#MAE=1.375793, time=0:00:46.381815 (K=1, steps=1)
#MAE=0.807655, time=0:04:40.349274 (K=10, steps=5)
#MAE=0.739657, time=0:04:20.913975 (k=10, steps=5, alpha=0.01, beta=0.01)
#MAE=0.740560 time=0:04:28.936162 (k=10, steps=5, alpha=0.001, beta=0.01)
