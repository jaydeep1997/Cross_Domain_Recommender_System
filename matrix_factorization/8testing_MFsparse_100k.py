#With the use of 3 lists, that contans UId, MId, rating
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
    A=[]
    B=[]
    C=[]
    for line in open(path+file):
        (user,movieid,rating,ts)=line.split('\t')
        A.append(int(user)-1)
        B.append(int(movieid)-1)
        C.append(float(rating))
        
    return A,B,C


def matrix_factorization(A, B, C, K, N, M, steps=5000, alpha=0.0002, beta=0.02):

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    print("makePQ_time=", datetime.now()-st)

    # print(P)
    # print(Q)

    Q = Q.T

    NN=len(A)

    for step in range(steps):
        e = 0
        for ii in range(NN) :
            i=A[ii]
            j=B[ii]
            eij=C[ii] - np.dot(P[i,:],Q[:,j])
            for k in range(K) :
                pik=P[i][k]
                qkj=Q[k][j]
                P[i][k] = pik + alpha * (2 * eij * qkj - beta * pik)
                Q[k][j] = qkj + alpha * (2 * eij * pik - beta * qkj)

            e = e + pow(C[ii] - np.dot(P[i,:],Q[:,j]), 2)
            for k in range(K) :
                e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break

    return P, Q.T


if __name__=='__main__':
    f=open("vary_beta","w")
    st=datetime.now()
    A,B,C = loadMovieLens(file="/u1.base")
    At,Bt,Ct = loadMovieLens(file='/u1.test')
    
    #print("load_time=", datetime.now()-st)
    #print("makeR_time=", datetime.now()-st)

    #print(trainPrefs[384][496])

    N=943
    M=1682
    K=10
    steps=5

    nP, nQ = matrix_factorization(A, B, C, K, N, M, steps, alpha=0.0002, beta=0.02)
    
    print("MF_time=", datetime.now()-st)
    
    # print(nP)
    # print(nQ)
    
    total_err=[]
    #calculating error (MAE)
    for ii in range(len(At)) :
        i=A[ii]
        j=B[ii]
        diff=fabs(np.dot(nP[i,:],nQ.T[:,j])-C[ii])
        total_err.append(diff)
    
    print(K,steps)
    print("MAE=%lf" % (sum(total_err)/len(total_err)))
    print("time=", datetime.now()-st)

#MAE=1.341527, time=0:00:24.973441 (k=2, steps=10)
#MAE=0.968972, time=0:00:44.945159 (K=10, steps=5)
#MAE=0.789122, time=0:00:43.349820 (k=10, steps=5, alpha=0.01, beta=0.01)
#MAE=0.799103, 0.802681 time=0:00:42.981212 (k=10, steps=5, alpha=0.001, beta=0.01)
#MAE=0.870521, time=0:01:39.292249 (k=5, steps=20)