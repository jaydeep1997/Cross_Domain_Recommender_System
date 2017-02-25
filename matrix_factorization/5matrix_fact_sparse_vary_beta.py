#varying beta with k=7 and steps=10
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
                    P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                    Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

                e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                for k in range(K) :
                    e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break


    return P, Q.T



if __name__=='__main__':
    f=open("vary_beta","w")
    st=datetime.now()
    trainPrefs = loadMovieLens(file="/u1.base")
    testPrefs = loadMovieLens(file='/u1.test')

    #print(trainPrefs[384][496])

    N=943
    M=1682
    K=7
    steps=10

    beta=0.01
    while beta<=0.1 :

        nP, nQ = matrix_factorization(trainPrefs, K, N, M, steps, 0.001, beta)
        
        sumErr=0;
        count=0
        #calculating error (MAE)
        for i in testPrefs :
                for j in testPrefs[i] :
                    diff=fabs(np.dot(nP[i,:],nQ.T[:,j])-testPrefs[i][j])
                    sumErr+=diff
                    count+=1
        
        print(K,steps)
        f.write("%lf %f\n" %(beta,sumErr/count))
        print("MAE=%lf %lf" %(beta,sumErr/count))
        print("time=", datetime.now()-st)
        beta+=0.01
