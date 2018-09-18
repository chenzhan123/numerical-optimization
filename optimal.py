import numpy as np
import linear_search

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

def only_linearcondition(dfun,d2fun,x0,A,maxiter=200,eps=1e-10):
    """
    min  f(x)
    s.t. Ax=b

    f(x[k]+p)=f[k]+f'[k]p+1/2*pf''(x)p
    A(x[k]+p)-b=Ax[k]+Ap-b=0

    KKT condition:
    L'[p,lambda]=f'[k]+f''[k]p-A*lambda=0
    Ax[k]+Ap-b=0

    the form of system of linear equations:
    [[f''[k]  A^T]  [[   p  ]  =  [[-f'[k]]
     [  A      0 ]]  [lambda]] =   [  0   ]]

    parameters
    ------------
    x0: initial point
        must satisfy constraint condition
    A: linear constraint condition
        must two-dimension.  e.g. np.array([[1,2,3],[4,1,2]])
    references
    -------------
    my notebook: 数值优化
    """
    x0=np.atleast_1d(x0)
    for i in range(maxiter):
        H = d2fun(x0)
        D, L = choleskey(H)
        row,col=A.shape
        l1=np.column_stack((L.dot(D).dot(L.T),A.T))
        l2=np.column_stack((A,np.zeros((row,row))))
        left_matrix=np.row_stack((l1,l2))
        r1=-dfun(x0)
        r2=np.zeros(row)
        right_matrix=np.concatenate((r1,r2))
        X=np.linalg.solve(left_matrix,right_matrix)
        p=X[:col]
        if p.T.dot(H).dot(p)<eps:
            break
        x0=x0+p
    return x0,i

def innerpoint(dfun,d2fun,x0,A,cfun,cdfun,cd2fun,eps=1e-10):
    """
    min  f(x)
    s.t. Ax=b
         Cx>=0
    ==>
    min  f(x)
    s.t. Ax-b=0
         C(x)s=1/t (s is variable)
         s>=0;C(x)

    KKT condition:
    L[p,y,z]=f[k]+f'[k]p+1/2*pf''[k]p+(C[k]+C'[k]p+1/2*pC''[k]p)(s+delta(s))+A^T(y+delta(y))
    L'[p,y,z]=f'[k]+f''[k]p+A(y+delta(y))+(C'[k]+C''[k]p)(s+delta(s))=0
                   =>  (f''[k]+sum(C''[k]*s))p + A^T(y+delta(y)) + C'[k]*delta(s)=-f'[k]-C'[k]s
    A(x[k]+p)=b    =>  A*p=0   (ensure both x[k] and x[k+1] satisfying: A(x[k]+p)=b )
    Ci(x[k]+p)(s+delta(s)=1/t*e  =>  (diag(C[k])+diag(C'[k]p))(s+delta(s))=1/t*e  =>

    the form of system of linear equations:
    [[f''[k]+C''[k]  A^T  C'[k]]    [[   p    ]    =  [[-f'[k]-C'[k]^Ts   ]
     [      A         0     0  ]    [y+delta(y)]   =   [         0        ]
     [   C'[k]^Ts     0    C[k]]]    [delta(s)]]   =   [      1/t-C[k]s   ]]

    parameters
    ------------
    x0: initial point; must satisfy constraint condition

    references
    -------------
    Stephen Boyed <convex optimization> 11.7
    my notebook: 数值优化

    """
    x0 = np.atleast_1d(x0)
    n=len(cdfun)
    s =np.empty(n)
    for i in range(n):
        s[i] = np.random.uniform(0, cfun[i](x0), size=1)
    t=1
    tau=n/t
    while tau>eps:
        tau = n/t
        H = d2fun(x0)
        D, L = choleskey(H)
        # l1=np.column_stack((L.dot(D).dot(L.T),A.T,np.zeros((A.shape[1],2*n))))
        dC=np.array(cdfun[0](x0))
        for i in range(1,n):
            dC=np.row_stack((dC,cdfun[i](x0)))
        C=[cfun[0](x0)]
        for i in range(1,n):
            C.append(cfun[i](x0))
        C=np.array(C)
        d2C = []
        for i in range(n):
            # print(np.atleast_2d(cd2fun[i](x0)*s[i]))
            d2C.append(np.atleast_2d(cd2fun[i](x0)*s[i]))
        CI=np.zeros((A.shape[1],A.shape[1]))
        for i in range(n):
            D1, L1 = choleskey(d2C[i])
            CI=CI+L1.dot(D1).dot(L1.T)
        dC=np.atleast_2d(dC)
        CI=np.atleast_2d(CI)
        l1=np.column_stack((L.dot(D).dot(L.T)+CI,A.T,dC.T))
        l2=np.column_stack((A,np.zeros((A.shape[0],A.shape[0]+n))))
        l3=np.column_stack((np.array([dC[i,:]*s[i] for i in range(n)]),np.zeros((n,A.shape[0])),np.diag(C)))
        left_matrix=np.row_stack((l1,l2,l3))
        r1 = -dfun(x0)-dC.T.dot(s)
        r2=np.zeros(A.shape[0])
        r3=np.ones(n)/t-np.diag(C).dot(s)

        right_matrix=np.concatenate((r1,r2,r3))
        X = np.linalg.solve(left_matrix, right_matrix)
        p=X[:len(x0)]
        y=X[len(x0):(len(x0)+A.shape[0])]
        delta_s=X[(len(x0)+A.shape[0]):]
        alpha=2
        con=np.array([-1])
        while (con<=0).all():
            alpha = alpha / 2
            con = []
            for i in range(n):
                try:
                    con.append(cfun[i](x0+alpha*p))
                except:
                    con = np.array([-1])
            con = np.array(con)
        while ((s + alpha * delta_s) <= 0).all():
            alpha = alpha / 2
        t=t*2
        x0 += alpha*p
        s += alpha*delta_s
    return x0

def barrier(dfun,d2fun,x0,A,cfun,cdfun,cd2fun,eps=1e-10):
    """
    min  f(x)
    s.t. Ax=b
         Cx>=0
    ==>
    min tf(x)+sum(-log(C(x)))
    s.t  Ax=b

    KKT condition:
    L'(x,y)=t*f'(x)-sum(1/C(x)*C'(x))+A^Ty=0
        => t(f'[k]+f''[k]p)+Ay-sum(1/C[k](C'[k]+C''[k]p))=0
    A(x+p)=b  =>  Ap=0

    [[tf''[k]+sum(1/C[k]*C''[k])  A^T ]   [[     p    ]   = [[sum(1/C[k]*C'[k])-tf'[k]]
     [              A              0  ]]   [y+delta(y)]]  =  [          0             ]]

    parameters
    ------------
    x0: initial point; must satisfy constraint condition

    references
    -------------------
    Stephen Boyed <convex optimization> 11.1 (formulation 11.14)
    """
    x0 = np.atleast_1d(x0)
    n = len(cdfun)
    t = 1
    while 1 / t > eps:
        dC = np.array(cdfun[0](x0))
        for i in range(1, n):
            dC = np.row_stack((dC, cdfun[i](x0)))
        dC = np.atleast_2d(dC)
        C = [cfun[0](x0)]
        for i in range(1, n):
            C.append(cfun[i](x0))
        C = np.array(C)
        d2C = []
        for i in range(n):
            d2C.append(np.atleast_2d(cd2fun[i](x0) / C[i]))
        CI = np.zeros((A.shape[1], A.shape[1]))
        for i in range(n):
            D1, L1 = choleskey(d2C[i])
            CI = CI + L1.dot(D1).dot(L1.T)
        CI = np.atleast_2d(CI)

        l1 = np.column_stack((t * d2fun(x0) + CI, A.T))
        l2 = np.column_stack((A, np.zeros((A.shape[0], A.shape[0]))))
        left_matrix = np.row_stack((l1, l2))
        r1 = np.array([dC[i, :] / C[i] for i in range(n)]).T
        r1 = r1.sum(1) - t * dfun(x0)
        r2 = np.zeros(A.shape[0])
        right_matrix = np.concatenate((r1, r2))
        X = np.linalg.solve(left_matrix, right_matrix)
        p = X[:len(x0)]
        x0 = x0 + p
        t *= 5
    return x0


