import numpy as np
from scipy.linalg.misc import norm


# __all__=['armijo_con','wolf_con','wolf_step','ArmijoBacktrack','quadra_itpola','quadra_itpola_3point']

# def onedim_search(fun,direction,x0):
#     alpha=1
#     if fun(x0+alpha*direction)<fun()

def armijo_con(fun,dfun,alpha,x,c1=0.5):
    """
    parameters
    ------------
    fun: input function
    dfun: gradient direction of the input function
    x: initial point
    w: the determined value of the fun

    references
    ------------
    http://www.math.mtu.edu/~msgocken/ma5630spring2003/lectures/lines/lines/node3.html"""
    x=np.atleast_1d(x)
    return fun(x-alpha*dfun(x))<=fun(x)-c1*alpha*np.dot(dfun(x),dfun(x))

def wolf_con(fun,dfun,alpha,x,c1=0.5,c2=0.7):
    """http://www.math.mtu.edu/~msgocken/ma5630spring2003/lectures/lines/lines/node3.html"""
    x = np.atleast_1d(x)
    return fun(x-alpha*dfun(x))<=fun(x)-c1*alpha*dfun(x)*dfun(x) and\
           -np.dot(dfun(x-alpha*dfun(x)),dfun(x))>=-c2*np.dot(dfun(x),dfun(x))

def arm_step(fun,dfun,x0,alpha=1,c1=0.5,p=None):
    """
    Parameters
    ----------
    p : gradient direction

    return
    ------
    alpha : a number of a step size
    """
    x0=np.atleast_1d(x0)
    if p is not None:
        p=np.atleast_1d(p)
    else:
        p=dfun(x0)
    while not fun(x0-alpha*p)<=fun(x0)-c1*alpha*np.dot(p,p):
        alpha=alpha*0.8
        # print('alpha2:',alpha)
    return alpha

def wolf_step(fun,dfun,x0,alpha=1e-7,c1=0.5,c2=0.7,p=None):
    """
    Parameters
    ----------
    p : gradient direction

    return
    ------
    alpha : a number of a step size
    """
    x0=np.atleast_1d(x0)
    if p is not None:
        p=np.atleast_1d(p)
    else:
        p=dfun(x0)
    while not -np.dot(dfun(x0-alpha*p),p)>=-c2*np.dot(p,p):
        alpha=alpha*1.5
        # print('alpha1:',alpha)
    while not fun(x0-alpha*p)<=fun(x0)-c1*alpha*np.dot(p,p):
        alpha=alpha*0.8
        # print('alpha2:',alpha)
    return alpha

def ArmijoBacktrack(fun,dfun,x0,maxiter=200,eps=1e-1,c1=0.5,p=None):
    """
    find an acceptable stepsize via backtrack under armijo rule

    parameters
    -----------
    fun:compute the value of objective function
    dfun:compute the gradient of objective function
    x0:the initial value of the paraments of fun
    w: the determined value of the fun
    maxiter:stop condition that the number of iteration
    eps: stop conditon that ||f'(x[k+1])||<eps
    c1:sufficient decrease Parameters
    """
    x0=np.atleast_1d(x0)
    try:
        for i in range(maxiter):
            if norm(dfun(x0),2)<eps:   #norm(np.atleast_1d(fun(x0)-fun(x0-alpha*dfun(x0))),2)<eta or
                break
            alpha=arm_step(fun,dfun,x0=x0,c1=c1,p=p)
            x0=x0-alpha*dfun(x0)
    except OverflowError:
        print('no minimum value')
    return x0,i

def WolfBacktrack(fun,dfun,x0,maxiter=200,eps=1e-1,c1=0.5,c2=0.7,p=None):
    """
    find an acceptable stepsize via backtrack under wolf rule

    parameters
    -----------
    fun:compute the value of objective function
    dfun:compute the gradient of objective function
    x0:the initial value of the paraments of fun
    w: the determined value of the fun
    maxiter:stop condition that the number of iteration
    eps: stop conditon that ||f'(x[k+1])||<eps
    c1:sufficient decrease Parameters
    c2:prevent slow Parameters
    """
    # eta:stop condition that |f(x[k+1])-f(x[k])|<eta
    # tal:stop condition that |x[k+1]-x[k]|<tal
    x0=np.atleast_1d(x0)
    try:
        for i in range(maxiter):
            if norm(dfun(x0),2)<eps:   #norm(np.atleast_1d(fun(x0)-fun(x0-alpha*dfun(x0))),2)<eta or
                break
            alpha=wolf_step(fun,dfun,x0=x0,c1=c1,c2=c2,p=p)
            x0=x0-alpha*dfun(x0)
    except OverflowError:
        print('no minimum value')
    return x0,i

def quadratic_interpolation(fun,dfun,x0,maxiter=200,tal=1e-1,c1=0.5):
    """
    references
    -------------
    PPT by 皱博
    """
    alpha=1
    x0 = np.atleast_1d(x0)
    for i in range(maxiter):
        if (np.abs(alpha*dfun(x0))<tal).all():
            break
        temp=alpha
        alpha=-np.dot(dfun(x0),dfun(x0))*alpha**2/(-np.dot(dfun(x0),dfun(x0))*alpha+fun(x0)-fun(x0-alpha*dfun(x0)))
        alpha=alpha/2
        if alpha<0: alpha=temp/2
        while not armijo_con(fun,dfun,alpha=alpha,x=x0,c1=c1):
            alpha/=1.5
            temp=alpha
            alpha = -np.dot(dfun(x0),dfun(x0))*alpha**2/(-np.dot(dfun(x0),dfun(x0))*alpha+fun(x0)-fun(x0-alpha*dfun(x0)))
            alpha = alpha / 2
            if alpha < 0: alpha = temp / 2
        x0 = x0 - alpha * dfun(x0)
    return x0,i

def quadratic_interpolation_3point(fun,dfun,x0,maxiter=200,tal=1e-1):
    """
    references
    -------------
    my notebook: 最优化
    """
    x0 = np.atleast_1d(x0)
    try:
        for i in range(maxiter):
            alpha = 1
            if (abs(alpha*dfun(x0))<tal).all():  #abs(fun(x0)-fun(x0-alpha*dfun(x0)))<eta and
                break
            if fun(x0-alpha*dfun(x0))>=fun(x0):
                start=alpha
                alpha = alpha / 2
                while fun(x0-alpha*dfun(x0))>=fun(x0):
                    alpha=alpha/2
                a=x0;b=x0-alpha*dfun(x0);c=x0-start*dfun(x0)
                x0=fun(a)*(c**2-b**2)+fun(b)*(a**2-c**2)+fun(c)*(b**2-a**2)
                x0=x0/(fun(a)*(c-b)+fun(b)*(a-c)+fun(c)*(b-a));x0=x0/2
            if fun(x0 - alpha * dfun(x0)) < fun(x0):
                start = alpha
                alpha = alpha * 2
                while fun(x0-alpha*dfun(x0))<=fun(x0):
                    alpha = alpha * 2
                a=x0;b=x0-start*dfun(x0);c=x0-alpha*dfun(x0)
                x0=fun(a)*(c**2-b**2)+fun(b)*(a**2-c**2)+fun(c)*(b**2-a**2)
                x0=x0/(fun(a)*(c-b)+fun(b)*(a-c)+fun(c)*(b-a));x0=x0/2
            # if temp=='inf':
            #     print('no minimum, may be maximum')
            #     break
    except OverflowError:
        print('no minimum value')
    return x0,i

# def trust_field(fun,dfun,d2fun,x0,maxiter=200,delta=1):
#     """
#     references
#     -------------
#     my notebook: 数值优化
#     """
#     x0 = np.atleast_1d(x0)
#     for i in range(maxiter):
#         alpha=np.linalg.norm(dfun(x0))**3/(dfun(x0).T.dot(d2fun(x0)).dot(dfun(x0)))
#         p=-alpha*dfun(x0)/np.linalg.norm(dfun(x0))
#         m=fun(x0)-np.linalg.norm(dfun(x0))**4/(dfun(x0).T.dot(d2fun(x0)).dot(dfun(x0)))/2
#         #update rho
#         rho=(fun(x0)-fun(x0+p))/(fun(x0)-m)
#         if rho<1/4:
#             delta=delta/4
#         if rho>3/4 and np.linalg.norm(p)>=delta:
#             p=-delta
#             delta=2*delta
#         #update x
#         if rho<1/4:
#             pass
#         else:
#             x0=x0+p
#     return x0,i


