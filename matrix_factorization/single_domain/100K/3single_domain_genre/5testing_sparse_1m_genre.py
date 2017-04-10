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

def loadGenre(path='.', file='/genrelist.dat'):

    G=np.zeros((3952,18))
    for line in open(path+file):
        (movieid,genre)=line.split('::')
        if(movieid != 'movieid') :
            l=[]
            l=genre.split(' ')
            for i in range(18) :
                l[i]=int(l[i])
            G[int(movieid)-1]=l
    return G


def matrix_factorization(R, genre, K, N, M, steps=50, alpha=0.0002, beta=0.02):

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    G = np.random.rand(18,K)
    
    print("makePQG_time=", datetime.now()-st)

    # print(P)
    # print(Q)

    Q = Q.T

    for step in range(steps):
        e = 0
        for i in R :
            for j in R[i] :
                count=0
                for i1 in range(18) :
                    if genre[j][i1]==1 :
                        count+=1;

                li=np.dot(genre[j],G)
                for i1 in range(K) :
                    li[i1]=li[i1]/count;    #average for jth item


                eij=R[i][j] - np.dot(P[i,:],Q[:,j]+li.T)
                for k in range(K) :
                    #P[i][k] = 5+5+5
                    #Q[k][j] = 5+5+5
                    P[i][k] = P[i][k] + alpha * (2 * eij * (Q[k][j] + li[k]) - beta * P[i][k])
                    Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
                    for i1 in range(18) :
                        if genre[j][i1]==1 :
                            G[i1][k] = G[i1][k] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])


    return P, Q.T, G



if __name__=='__main__':
    st=datetime.now()
    trainPrefs = loadMovieLens(file="/Train.dat")
    testPrefs = loadMovieLens(file='/Test.dat')
    genre = loadGenre(file='/genrelist.dat')

    # for i in range(3952):
    #     for j in range(18) :
    #         print(genre[i,j],end=' ')
    #     print("\n")
    
    print("load_time=", datetime.now()-st)
    print("makeR_time=", datetime.now()-st)

    N=6040
    M=3952
    K=5
    steps=5



    nP, nQ, nG = matrix_factorization(trainPrefs, genre, K, N, M, steps, alpha=0.0002, beta=0.02)
    
    print("MF_time=", datetime.now()-st)
    
    # print(nP)
    # print(nQ)
    
    total_err=[]
    #calculating error (MAE)
    for i in testPrefs :
            for j in testPrefs[i] :
                count=0
                for i1 in range(18) :
                    if genre[j][i1]==1 :
                        count+=1;

                li=np.dot(genre[j],nG)
                for i1 in range(K) :
                    li[i1]=li[i1]/count;    #average for jth item

                diff=fabs(np.dot(nP[i,:],nQ.T[:,j]+li.T)-testPrefs[i][j])
                total_err.append(diff)
    
    print(K,steps)
    print("MAE=%lf" % (sum(total_err)/len(total_err)))
    print("time=", datetime.now()-st)

#MAE=0.933934,(k=1, steps=1)
#MAE=0.968972, time=0:00:44.945159 (K=10, steps=5)
