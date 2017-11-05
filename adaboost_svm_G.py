
from scipy.optimize import minimize
import numpy as np
from gurobipy import *


def GaussianKernel(X,X1,beta):
    tx=np.sum(np.transpose(X)**2,0)
    tx1=np.sum(np.transpose(X1)**2,0)
    D=np.zeros((len(X),len(X1)))
    for i in range(len(X)):
        for j in range(len(X1)):
            D[i,j]=tx[i]+tx1[j]
    D=D-2*np.dot(X,np.transpose(X1))
    K=np.exp(-beta*D)
    return K

def quadopt(alpha0,weights,C,Ytrain,H):
    model=Model('SVM')
    xIdx=[i for i in range(len(H))]
    x={}
    for i in range(len(xIdx)):
        x[i]=model.addVar(lb=0,ub=weights[i]*C)
    model.addConstr(sum([x[i]*Ytrain[i] for i in range(len(x))])==0,'constraint1')

    obj=0
    for i in range(len(H)):
        for j in range(len(H)):
            obj=obj-0.5*x[i]*x[j]*H[i,j]
        obj=obj+x[i]
    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    alpha=np.array(model.x)
    return alpha
#check results
#for v in model.getVars():
#    print('%s %g' % (v.varName, v.x))
#model.objVal


def AdaBoost_SVM(nbRun,C,KT,Kval,Ktest,H,Ytrain,Yval,Ytest):
    n=KT.shape[0]
    nbVal=Kval.shape[0]
    nbTest=Ktest.shape[0]
    alphaV=np.zeros((nbRun,1))
    GM=np.zeros((n,nbRun))
    GMVal=np.zeros((nbVal,nbRun))
    miserrorVal=np.zeros(nbRun,)
    GMTest=np.zeros((nbTest,nbRun))
    miserrorTest=np.zeros(nbRun)
    
    alpha0=np.ones(n)
    weights=np.ones(n)
    
    for i in range(nbRun):
        alpha=quadopt(alpha0,weights,C,Ytrain,H)
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





