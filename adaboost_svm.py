
from scipy.optimize import minimize#use scipy minimization package
import numpy as np



def GaussianKernel(X,X1,beta): #code gaussian kernal, K(X,X1), bandwidth beta
    tx=np.sum(np.transpose(X)**2,0)
    tx1=np.sum(np.transpose(X1)**2,0)
    D=np.zeros((len(X),len(X1)))
    for i in range(len(X)):
        for j in range(len(X1)):
            D[i,j]=tx[i]+tx1[j]
    D=D-2*np.dot(X,np.transpose(X1))
    K=np.exp(-beta*D)
    return K



def func(x,H):#the quadratic objective function with coefficient matrix H
    f=0
    for i in range(len(x)):
        f=H[i,0]*x[i]
        for j in range(len(x)):
            f=f+0.5*H[i,j]*x[i]*x[j]
    f=f-sum(x)
    return f



def quadopt(alpha,weights,C,Ytrain,H):#code the optimization problem
    cons = [{'type': 'eq', #Equality constraint
          'fun' : lambda x: sum(x*Ytrain),
        }]
    bnds=tuple([(0,weights[i]*C) for i in range(len(weights))])#bound constraint
    res=minimize(func,alpha,method='SLSQP',jac=fun_derive,args=H,bounds=bnds,constraints=cons)#minimization problem
    return res.x#return the solution
    



def fun_derive(x,H):#code the derivative of the objective function wrt x
    jac=[]#jacobian
    for i in range(len(x)):
        jac_i=0
        for j in range(len(x)):
            jac_i=jac_i+0.5*H[i,j]*x[j]#derivation of obj wrt varialbe i
        jac_i=jac_i-1
        jac.append(jac_i)
    return jac



def AdaBoost_SVM(nbRun,C,KT,Kval,Ktest,H,Ytrain,Yval,Ytest):#the adaboost algorithm
    n=KT.shape[0]#nb training samples
    nbVal=Kval.shape[0]#nb validation samples
    nbTest=Ktest.shape[0]#nb test samples
    alphaV=np.zeros((nbRun,1))#alpha value of adaboost
    GM=np.zeros((n,nbRun))
    GMVal=np.zeros((nbVal,nbRun))
    miserrorVal=np.zeros(nbRun,)
    GMTest=np.zeros((nbTest,nbRun))
    miserrorTest=np.zeros(nbRun)
    
    alpha0=np.ones(n)
    weights=np.ones(n)
    
    for i in range(nbRun):
        alpha=quadopt(alpha0,weights,C,Ytrain,H)#dual problem quadratic programming
        Ypred=np.sign(np.dot(KT,(alpha*Ytrain)))
        YpredVal=np.sign(np.dot(Kval,(alpha*Ytrain)))
        YpredTest=np.sign(np.dot(Ktest,(alpha*Ytrain)))
        miserror=1-np.sum(Ypred==Ytrain)/len(Ytrain)
        
        wSample=np.where(Ypred!=Ytrain)
        wError=np.sum(weights[wSample])/np.sum(weights)
        alphaV[i]=np.log((1-wError)/wError)
        GM[:,i]=Ypred*alphaV[i]
        GMVal[:,i]=YpredVal*alphaV[i]
        GMTest[:,i]=YpredTest*alphaV[i]
        GVal=np.sign(np.sum(GMVal,1))
        GTest=np.sign(np.sum(GMTest,1))
        miserrorVal[i]=1-np.sum(GVal==Yval)/len(Yval)
        miserrorTest[i]=1-np.sum(GTest==Ytest)/len(Ytest)
        weights[wSample]=weights[wSample]*np.exp(alphaV[i])#update weights
    Gpred=np.sign(np.sum(GM,1))
    GpredVal=np.sign(np.sum(GMVal,1))
    GpredTest=np.sign(np.sum(GMTest,1))
    #miserrorFin=1-np.sum(Gpred==Ytrain)/len(Ytrain)
    return (Gpred,GpredVal,GpredTest)




