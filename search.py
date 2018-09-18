import linear_search
from scipy.linalg.misc import norm
import numpy as np

def choleskey(matrix,delta=1e-3):
    A=np.atleast_2d(matrix)
    row,col=A.shape
    D=np.zeros(row);D=np.array(np.diag(D))
    L=np.ones(row);L=np.array(np.diag(L))
    while((np.diag(D)<=0).any()):
        for j in range(col):
            temp=0
            for k in range(0,j):
                temp+=D[k,k]*L[j,k]**2
            if A[j, j]-temp>0:
                D[j,j]=(A[j, j]-temp)
            else:
                A=A+(delta-(A[j, j]-temp))*np.array(np.diag(np.ones(row)))
                break
            for i in range(j+1,row):
                temp=0
                for k in range(0, j):
                    temp += D[k,k] * L[j, k] *L[i, k]
                L[i,j]=(A[i,j]-temp)/D[j,j]
    return D,L

def Newton(dfun,d2fun,x0,maxiter=200,epsilon=1e-10):
    """
    d2fun needed to return a array of 2-dim
    """
    x0 = np.atleast_1d(x0)
    for i in range(maxiter):
        H=d2fun(x0)
        D,L=choleskey(H)
        b1=np.linalg.solve(L,-dfun(x0))
        b2=np.linalg.solve(D,b1)
        p=np.linalg.solve(L.T,b2)
        if abs(p.dot(H).dot(p))<epsilon:   #abs(fun(x0,w)-fun(x0+alpha*v,w))<eta and
            break
        x0=x0+p
    return x0,i

def Conjugate_dir(fun,dfun,x0,Linear_Search,maxiter=200,tal=1e-10):
    x0 = np.atleast_1d(x0)
    v=-dfun(x0)
    for i in range(maxiter):
        alpha=Linear_Search(fun,dfun=dfun,x0=x0,p=-v)
        if (abs(v)<tal).all():
            break
        denom=norm(np.atleast_1d(dfun(x0)), 2)
        x0=x0+alpha*v
        num=norm(np.atleast_1d(dfun(x0)), 2)
        v=-dfun(x0)-num/denom*v
    return x0,i

def DFP(fun,dfun,x0,Linear_Search,maxiter=200,epsilon=1e-10,tal=1e-3,c1=0.5,c2=0.8):
    """
    references
    -----------------
    李航 <统计学习方法>
    """
    x0=np.atleast_1d(x0)
    G=np.eye(len(x0));g=np.atleast_1d(dfun(x0))
    for i in range(maxiter):
        if norm(dfun(x0),2)<epsilon:
            break
        p=np.dot(G,g)     #p is -p in Li Hang's book
        alpha=Linear_Search(fun,dfun,x0,c1=c1,p=p)
        x=x0
        x0=x0-alpha*p
        delta=x0-x;y=dfun(x0)-dfun(x)
        if all(np.abs(delta)<tal) or all(np.abs(y)<tal):
            break
        Q=-np.dot(G,np.outer(y.T,np.dot(y.T,G)))/np.dot(np.dot(y.T,G),y)
        G=G+np.outer(delta,delta.T)/np.dot(delta.T,y)+Q
        g = dfun(x0)
    return x0,i


def BFGS(fun,dfun,x0,Linear_Search,maxiter=200,epsilon=1e-10,c1=0.5,c2=0.8):
    """
    references
    -----------------
    李航 <统计学习方法>
    Stephen Boyed <convex optimization> (only refer to stop condtion:the newton decrement)
    """
    x0=np.atleast_1d(x0)
    B=np.eye(len(x0));g=np.atleast_1d(dfun(x0))
    for i in range(maxiter):
        if norm(g.dot(B).dot(g))<epsilon:
            break
        p=np.linalg.solve(B,g)    #p is -p in Li Hang's book
        alpha=Linear_Search(fun,dfun,x0,c1=c1,p=p)
        x=x0
        x0=x0-alpha*p
        delta=x0-x;y=dfun(x0)-dfun(x)
        Q=-np.dot(B,np.outer(delta,np.dot(delta.T,B)))/np.dot(delta.T,np.dot(B,delta))
        B=B+np.outer(y,y.T)/np.dot(y.T,delta)+Q
        g=dfun(x0)
    return x0,i

