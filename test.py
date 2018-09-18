from optimal import *
from search import *
from linear_search import *
import numpy as np
import pandas as pd
data2000=pd.read_excel("data\\2000.xlsx").dropna().loc[:,[' Rank',' Est. Monthly Sales']]
data2001=pd.read_excel("data\\2001.xlsx").dropna().loc[:,[' Rank',' Est. Monthly Sales']]
data=pd.concat([data2000,data2001],axis=0)
data=data.drop_duplicates()
data=data.sort_index(by=[' Rank',' Est. Monthly Sales'])
data['index']=0
for i in range(len(data) - 1):
    if data[' Est. Monthly Sales'].iloc[i] - data[' Est. Monthly Sales'].iloc[i+1] <= 0:
        data['index'].iloc[i]=np.nan

data.dropna(inplace=True)
data[' Est. Monthly Sales']=data[' Est. Monthly Sales']/30
x=np.concatenate((data[' Rank'],data[' Est. Monthly Sales']))


# f=ax^b*e^(cx)  <=>  lnf=(lna)+blnx+cx
# df=sum((lny-lnf)[-1,-lnx,-x])
def f(w):
    return sum((np.log(x[int(len(x) / 2):]) - (w[0] + w[1] * np.log(x[:int(len(x) / 2)]) + w[2] * x[:int(len(x) / 2)]))**2)/2

def df(w):
    lny_lnf=np.log(x[int(len(x)/2):])-(w[0]+w[1]*np.log(x[:int(len(x)/2)])+w[2]*x[:int(len(x)/2)])
    v1=-sum(lny_lnf)
    v2=-sum(lny_lnf*np.log(x[:int(len(x)/2)]))
    v3=-sum(lny_lnf*x[:int(len(x)/2)])
    return np.array([v1,v2,v3])

def d2f(w):
    # lny_lnf=np.log(x[int(len(x)/2):])-(w[0]+w[1]*np.log(x[:int(len(x)/2)])-w[2]*x[:int(len(x)/2)])
    v11=len(x[:int(len(x)/2)])
    v21=sum(np.log(x[:int(len(x)/2)]))
    v31=sum(x[:int(len(x)/2)])
    v22=sum(np.log(x[:int(len(x)/2)])**2)
    v32=sum(x[:int(len(x)/2)]*np.log(x[:int(len(x)/2)]))
    v33=sum(x[:int(len(x)/2)]**2)
    H=np.array([[v11,v21,v31],[v21,v22,v32],[v31,v32,v33]])
    return H+0.001*np.array(np.diag(np.ones(3)))

###########################################
######   optimal with unconstraint   ######
###########################################
#least squares solution:[ 5.89035124e+00, -2.44169188e-01, -1.48217833e-04]
# 1.now we test linear search methods
ArmijoBacktrack(f,df,[1,-2.0,-1.0],maxiter=1000)  #[ 1.05450461, -1.7427339 ,  0.01266699]
WolfBacktrack(f,df,[1,-2.0,-1.0],maxiter=1000)   #[ 1.02523163, -1.87978264,  0.013549  ]
quadratic_interpolation(f,df,[1,-2.0,-1.0],maxiter=200,tal=1e-1,c1=0.5)    #[ 1.00085528, -1.99405267,  0.01417695]
quadratic_interpolation_3point(f,df,[1,-2.0,-1.0],maxiter=200,tal=1e-1)    #[ 4.44894736e+00,  5.55648948e-02, -7.05302823e-04]

# hard to converse because the condition number of hessian matrix is large
# 2.now we test unlinear search methods
Newton(df,d2f,[1,-2.0,1.0],maxiter=200,epsilon=1e-10)      #[ 5.89035121e+00, -2.44169183e-01, -1.48217844e-04]
Conjugate_dir(f,df,[1,-2.0,1.0],wolf_step,maxiter=2000)  #[ 0.99926245, -2.00514727,  0.11567255]
DFP(f,df,[1,-2.0,1.0],wolf_step,maxiter=200)   #[ 5.89037624e+00, -2.44174113e-01, -1.48226787e-04]
BFGS(f,df,[1,-2.0,1.0],arm_step,maxiter=200)   #[ 5.89035124e+00, -2.44169188e-01, -1.48217833e-04]

# 3.now we use another function that is simple to test
# least squares solution:[ 5.89035124e+00, -2.44169188e-01, -1.48217833e-04]
# y=x1^2+x2^2+3
f=lambda x:(x[0]**2+x[1]**2+3)
df=lambda x:(np.array([2*x[0],2*x[1]]))
d2f=lambda x:(np.array([[2,0],[0,2]]))

ArmijoBacktrack(f,df,[1,-2.0],maxiter=1000)  #array([ 0.00591011, -0.01182021]), 3
WolfBacktrack(f,df,[1,-2.0],maxiter=1000)   #array([ 0.0179664, -0.0359328]), 7
quadratic_interpolation(f,df,[1,-2.0],maxiter=200,tal=1e-1,c1=0.5)    #array([0., 0.]), 1
quadratic_interpolation_3point(f,df,[1,-2.0],maxiter=200,tal=1e-1)    #array([-0.,  0.]), 1


#########################################
######   optimal with constraint   ######
#########################################
"""
constraint:
w1+w2+w3=0
"""
# A=np.array([[1,1,1]])
# # only contain linear condition
# only_linearcondition(df,d2f,[1,-2.0,1.0],A)   #[-1.19138238,  1.19399383, -0.00261145]

"""
constraint:
w1+w2+w3=0
3w1+2w2+2w2=1
w2^2+w3^2-1>=0
w1+w2+w3^2+2>=0
"""
A=np.array([[1,1,1],[3,2,2]])    #equal constraint condition
cfun=(lambda w:w[1]**2+w[2]**2-1,lambda w:w[0]+w[1]+w[2]**2+2,)    #inequal constraint condition
cdfun=(lambda w:np.array([0,2*w[1],2*w[2]]),lambda w:np.array([1,1,2*w[2]]),)   #gradient of inequal constraint condition
cd2fun=(lambda w:np.array([[0,0,0],[0,2,0],[0,0,2]]),lambda w:np.array([[0,0,0],[0,0,0],[0,0,2]]),)
# apply innerpoint method
innerpoint(df,d2f,[1,-2.0,1.0],A,cfun,cdfun,cd2fun,eps=1e-10)  # [ 1. , -1.00840461,  0.00840461]
# apply barrier method
barrier(df,d2f,[1.0,-2.0,1.0],A,cfun,cdfun,cd2fun,eps=1e-10)  #[ 1.        , -1.00840461,  0.00840461]


