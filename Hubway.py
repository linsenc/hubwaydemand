
#The bikesharing system "Hubway" short term demand prediction problem
#Linsen Chong

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC

from adaboost_svm import *# import functions written for adaboost SVM

folder='C:/Hubway/Project/'#Please change to local folder
stationfile='stationsub3.csv'#a subset of snapshot dataset
weatherfile_may='weatherMay.csv'
weatherfile_june='weatherJune.csv'
weatherfile_july='weatherJuly.csv'



def preprocess(stationData,weatherData):#preprocessing function
    #three steps:1) process station snapshot data 2) process weather data 3) and merge
    st_may=stationData[stationData['month']==5]#may data
    st_june=stationData[stationData['month']==6]#june data
    st_july=stationData[stationData['month']==7]#july data
    st_july_sub=st_july[st_july['day']<30]#until July 29th
    st_may_sub=st_may[st_may['day']>5]#from May 6th
    stdata=st_may_sub.append([st_june,st_july_sub])
    
    stdata['time']=stdata['time'].map(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    stdata['quarter']=stdata['time'].map(lambda x: int(x.minute/15)+1)
    stdata['hour']=stdata['time'].map(lambda x: x.hour)
    #time_station_ID: a unique ID to specify the index of a station at a quarter (of an hour) of a given day
    stdata['time_station_ID']=stdata['month']*10000000+stdata['day']*100000+stdata['hour']*1000+stdata['quarter']*100+stdata['station_id']
    stdata['weekday']=stdata['time'].map(lambda x: x.weekday())
    stgroupby=stdata.groupby(['time_station_ID'])#use grouby to figure out at each time_station_ID, the average number of bikes
    
    stdataQ=stgroupby['nbBikes'].mean().reset_index() #dataframe at the time scale of quarter   
    stdataQ['weekday']=stgroupby['weekday'].mean().values
    stdataQ['nbEmptyDocks']=stgroupby['nbEmptyDocks'].mean().values
    stdataQ['capacity']=stgroupby['capacity'].mean().values
    stdataQ['nbBikes']=stdataQ['nbBikes']/stdataQ['capacity']#normalize bike data
    stdataQ['nbBikes']=stdataQ['nbBikes'].fillna(0)
    
    
    allstationID=np.unique(stdataQ['time_station_ID']%100) #The IDs of all the stations
    
    allstationData=stdataQ
    allstationData['station']=allstationData['time_station_ID']%100#from the unique station-time ID, get the station
    allstationData['month']=allstationData['time_station_ID']//10000000#get the month
    allstationData['quarter']=allstationData['time_station_ID']//100%10#get the quarter
    allstationData['hour']=allstationData['time_station_ID']//1000%100#get the hour
    allstationData['day']=allstationData['time_station_ID']//100000%100#get the day

    #store all data in a dictionary: allstationT, where the key of the dictory is the stationID, this step is useful for doinig 
    #stationbased analysis
  
    allstationT={}
    for i in allstationID:
        stationData=allstationData[allstationData['station']==i]
        stationDataT=stationData[(stationData['hour']!=16)|(stationData['quarter']!=1)] #remove the first quarter
        stationDataT=stationDataT[(stationDataT['hour']!=18)|(stationDataT['quarter']!=4)]#remove the last quarter data  
        stationDataPrev=stationData[(stationData['hour']!=18)|(stationData['quarter']<3)]['nbBikes'].values#number of bikes of the previous quarter
        stationDataNext=stationData[(stationData['hour']!=16)|(stationData['quarter']>2)]['nbBikes'].values#number of bikes of the next quarter
        stationDataT['prevBikes']=stationDataPrev#add feature prevBikes
        stationDataT['nextBikes']=stationDataNext#add feature bikes of the next intervals
        allstationT[i]=stationDataT
        
        
    #add feature: number of bikes at all stations, i.e., column nbBikes_stn_j added below is the number of bikes at the station j at the quarter
    for i in allstationID:
        allstationT[i]['hourID']=allstationT[i]['time_station_ID']//1000#construct hourID feature to merge with weather data
        for j in allstationID:
            colName='nbBikes_stn_'+str(j)
            allstationT[i][colName]=allstationT[j]['nbBikes'].values
    
    #Now preprocess weather data
    
    #Get useful weather data
    weatherTot=weatherData[['Date','Time','Visibility','Rain','Drizzle','Mist','Fog','DryBulbFarenheit','RelativeHumidity','WindSpeed']]
    weatherTot=weatherTot.fillna(0)#fill the NA due to the data schema
    weatherTot['hour']=weatherTot['Date']*100+weatherTot['Time']//100#use a unique ID to get the hour
    weatherHour=weatherTot[(weatherTot['Time']//100>15)&(weatherTot['Time']//100<19)]#from 15 to 19 pm
    weatherHour=weatherHour[(weatherHour['Date']>20120505)&(weatherHour['Date']<20120730)] #extract data from the days of interest
    
    #groupby the unique ID of hour
    weatherHourGroupby=weatherHour.groupby(['hour'])
    weatherHourNew=weatherHourGroupby['Rain'].max().reset_index()#dataframe at the time scale of an hour
    weatherHourNew['Drizzle']=weatherHourGroupby['Drizzle'].max().values
    weatherHourNew['Mist']=weatherHourGroupby['Mist'].max().values
    weatherHourNew['Fog']=weatherHourGroupby['Fog'].max().values
    weatherHourNew['Temp']=weatherHourGroupby['DryBulbFarenheit'].mean().values#temperature
    weatherHourNew['Humid']=weatherHourGroupby['RelativeHumidity'].mean().values#humidity
    weatherHourNew['Visibility']=weatherHourGroupby['Visibility'].mean().values#visibility
    weatherHourNew['WindSpeed']=weatherHourGroupby['WindSpeed'].mean().values#wind speed
    
    #normalize Temp
    normalizeCol=['Temp','Humid','Visibility','WindSpeed']
    scaler=MinMaxScaler()
    scaled_values=scaler.fit_transform(weatherHourNew[normalizeCol])
    weatherHourNew[normalizeCol]=pd.DataFrame(scaled_values,columns=normalizeCol)
    weatherHourNew['hourID']=weatherHourNew['hour']%1000000
    weatherHourNew.head()  
    
    #step 3: merge station snapshoot data with weatherData
    for i in allstationID:
        allstationT[i]=allstationT[i].merge(weatherHourNew,left_on='hourID',right_on='hourID')#merge snapshoot with data
        
        #add feature holiday
        allstationT[i]['holiday']=(((allstationT[i]['month']==5) & (allstationT[i]['day']==28) )|((allstationT[i]['month']==7) & (allstationT[i]['day']==4)))+0
        allstationT[i]['wday']=allstationT[i]['weekday'].map(lambda x: 1 if x!=0 and x!=6 else 0)#add binary feature to tell weekday/weekend
        allstationT[i]['daytime']=allstationT[i]['hour_x'].map(lambda x: 1 if x>0 and x<19 else 0)#add feature daytime
    
        #add normalized version of capacity
        scaler=MinMaxScaler()
        allstationT[i]['capacityN']=np.array(scaler.fit_transform(allstationT[i][['capacity']]))
        
    allstationT=data_clean_all_ohe(allstationT,allstationID)
    
    return allstationT
    



def data_clean_all_ohe(allstationT,allstationID):#preprocess categorical variables
    nbstation=len(allstationT)
    for i in allstationID:
        df_dummy=pd.concat([pd.get_dummies(allstationT[i]['quarter'],prefix='quarter'),pd.get_dummies(allstationT[i]['hour_x'],prefix='hour')],axis=1)
        allstationT[i]=pd.concat([allstationT[i],df_dummy],axis=1)
    return allstationT



def splitdata(allstationT,threshold,allstationID):#split training and testing data
    allstationTrain={}
    allstationTest={}
    allpredictTrain={}
    allpredictTest={}
    
    
    for i in allstationID:
        allstationTrain[i]=allstationT[i].iloc[:threshold]
        allstationTest[i]=allstationT[i].iloc[threshold:]
        allpredictTrain[i]=allstationT[i].iloc[:threshold]['nextBikes']
        allpredictTest[i]=allstationT[i].iloc[threshold:]['nextBikes']
    return allstationTrain,allstationTest,allpredictTrain,allpredictTest



class RegressionModel(): #class of regression model
    def __init__(self, RGmethod,params):
        
        self.method = RGmethod #a regression model
        self.params = params #model parameters 
        self.model=[] #the final model is saved here
        
        # extra variable to store the trained model
        self._train_preds = []#save predictions of the training data
        self._test_preds = []#save predictions of the test data
        self._train_MSE =0#the MSE on training dataset
        self._test_MSE=0#the MSE on testing set
        
    def fit(self,X,Y):
        kr = GridSearchCV(self.method, cv=5, param_grid=self.params)#use gridsearch to find the best parameters for cross validation
        self.model=kr.fit(X,Y)#fit the model
        self._train_preds=self.model.predict(X)#make estimates for training set
        self._train_MSE=np.sum((self._train_preds-Y)**2)/len(Y)#MSE
        return self
    
    def predict(self,Xtest,Ytest):
        Ypred=self.model.predict(Xtest)
        self._test_preds=self.model.predict(Xtest)#prediction for test set
        self._test_MSE=np.sum((self._test_preds-Ytest)**2)/len(Ytest)#MSE
        return self
    
    def getPerformanceMetric(self):
        return self._train_MSE,self._test_MSE
    



class ClassificationModel(): #class of (non-adaboost) classification model
    def __init__(self, clfmethod,params):
        
        self.method = clfmethod #a classification model
        self.params = params #model parameters
        self.model=[] #the final model
        
        # extra variable to store the trained model
        self._train_preds = []
        self._test_preds = []
        self._train_E =0
        self._test_E=0
        
    def fit(self,X,Y): #fit the model
        clf = GridSearchCV(self.method, cv=5, param_grid=self.params)
        self.model=clf.fit(X,Y)
        self._train_preds=self.model.predict(X)
        self._train_E=1-np.sum(self._train_preds==Y)/len(Y)
        return self
    
    def predict(self,Xtest,Ytest): #prediction on test data
        Ypred=self.model.predict(Xtest)
        self._test_preds=self.model.predict(Xtest)
        self._test_E=1-np.sum(self._test_preds==Ytest)/len(Ytest)
        return self
    
    def getPerformanceMetric(self):
        return self._train_E,self._test_E
    



class AdaboostModel(): #class of adaboost model
    def __init__(self, nbRuns,beta,model,C):
        
        self.model=model
        self.nbRuns=nbRuns
        self.beta=beta
        self.C=C
        #self.K=None
        #self.Kval=None
        #self.Ktest=None
        
        # extra variable to store the trained model
        self._train_preds = []#prediction on training data
        self._val_preds = []#prediction validation data        
        self._test_preds = []# prediction test data
        self._train_E =0#misclassification error, training
        self._val_E =0#misclassifiction error, validation
        self._test_E=0#misclassification error, test
        
    def fit_predict(self,X,Y,Xval,Yval,Xtest,Ytest):
        if self.model=='kernel':#adaboost gaussian kernel
            K=GaussianKernel(X,X,self.beta)
            Kval=GaussianKernel(Xval,X,self.beta)
            Ktest=GaussianKernel(Xtest,X,self.beta)
            #vi=np.dot(np.reshape(Y,(len(Y),1)),np.reshape(Y,(1,len(Y))))
        if self.model=='linear':#adaboost linear kernel
            K=np.dot(X,np.transpose(X))
            Kval=np.dot(Xval,np.transpose(X))
            Ktest=np.dot(Xtest,np.transpose(X))
        vi=np.dot(np.reshape(Y,(len(Y),1)),np.reshape(Y,(1,len(Y))))
        H=K*vi
        self._train_preds,self._val_preds,self._test_preds=AdaBoost_SVM(self.nbRuns,self.C,K,Kval,Ktest,H,Y,Yval,Ytest)
        self._train_E=1-np.sum(self._train_preds==Y)/len(Y)
        self._val_E=1-np.sum(self._val_preds==Yval)/len(Yval)
        self._test_E=1-np.sum(self._test_preds==Ytest)/len(Ytest)
        return self
    
    #def predict(self,Xtest,Ytest):
    #    AdaBoost_SVM(self.nbRuns,KT,Kval,Ktest)
    #    Ypred=self.model.predict(Xtest)
    #    self._test_preds=self.model.predict(Xtest)
    #    self._test_E=1-np.sum(self._test_preds==Ytest)/len(Ytest)
    #    return self
    
    def getPerformanceMetric(self):
        return self._train_E,self._val_E,self._test_E
    



#regression models building, training and testing

def regression_mdls(allstationTrain,allstationTest,allstationID,featureCol):
    nbstation=len(allstationID)#number of stations
    MSETrain_ridge_kernel=np.zeros(nbstation)#MSE ridge regression with Gaussian Kernel, training data
    MSETrain_ridge=np.zeros(nbstation)#MSE ridge regression, training data
    MSETrain_lasso=np.zeros(nbstation)#MSE lasso regression, training data
    MSETest_ridge_kernel=np.zeros(nbstation)#MSE test data
    MSETest_ridge=np.zeros(nbstation)
    MSETest_lasso=np.zeros(nbstation)

    krg_model=[]
    rg_model=[]
    lasso_model=[]
    param_grid_krg = {"alpha": [1e-4, 1e-3, 1e-2,1e-1, 1, 10,100], #the parameters of the ridge regression with Gaussian kernel
                  "kernel": ['rbf'],
                  "gamma":[1e-4, 1e-3, 1e-2,1e-1, 1, 10,100]}
    param_grid_rg={"alpha":[1e-4, 1e-3, 1e-2,1e-1, 1, 10,100]}#parameters of the ridge regression model
    param_grid_lasso={"alpha":[1e-4, 1e-3, 1e-2,1e-1, 1, 10,100]}#parameters of the lasso regression model
    
    for i in range(len(allstationID)):
    #for i in range(1):
        XtrainRG=np.array(allstationTrain[allstationID[i]][featureCol])#
        YtrainRG=np.array(allstationTrain[allstationID[i]]['nextBikes'])
        XtestRG=np.array(allstationTest[allstationID[i]][featureCol])#
        YtestRG=np.array(allstationTest[allstationID[i]]['nextBikes'])
    
        krg = RegressionModel(KernelRidge(), param_grid_krg)#ridge regression with Gaussian Kernel
        krg=krg.fit(XtrainRG, YtrainRG)
        krg=krg.predict(XtestRG,YtestRG)
        krg_model.append(krg)
        MSETrain_ridge_kernel[i],MSETest_ridge_kernel[i]=krg.getPerformanceMetric()
    
        rg = RegressionModel(Ridge(), param_grid_rg)#ridge regression with linear kernel
        rg=rg.fit(XtrainRG, YtrainRG)
        rg=rg.predict(XtestRG,YtestRG)
        rg_model.append(rg)
        MSETrain_ridge[i],MSETest_ridge[i]=rg.getPerformanceMetric()
    
        rlasso = RegressionModel(Lasso(), param_grid_lasso)#lasso regression
        rlasso=rlasso.fit(XtrainRG, YtrainRG)
        rlasso=rlasso.predict(XtestRG,YtestRG)
        lasso_model.append(rlasso)
        MSETrain_lasso[i],MSETest_lasso[i]=rlasso.getPerformanceMetric()
    return MSETest_ridge_kernel,MSETest_ridge,MSETest_lasso
    


#classification model building
def classification_mdls(allstationTrain,allstationTest,allstationID,featureCol):
    nbstation=len(allstationID)
    ETrain_SVM=np.zeros(nbstation)#misclassification error, training data SVM
    ETest_SVM=np.zeros(nbstation)#misclassificaiton error, test data SVM
    ETrain_RDforest=np.zeros(nbstation)#misclassication error, random forest
    ETest_RDforest=np.zeros(nbstation)
    
    SVM_model=[]
    RDforest_model=[]
    #parameters for SVM
    param_grid_SVM = {"C": [1e-5, 1e-4, 1e-3,1e-2,1e-1, 1, 10,100,1000,10000,100000],
                      "kernel": ['rbf'],
    "gamma":[1e-5, 1e-4, 1e-3,1e-2,1e-1, 1, 10,100,1000,10000,100000]}
    #parameters for random forest
    param_grid_rd={'n_estimators':range(50,400,50),'min_samples_leaf':range(2,20,5)}
    
    for i in range(len(allstationID)):
    #for i in range(1):
        Xtrain=np.array(allstationTrain[allstationID[i]][featureCol])#
        Ytrain=np.array(allstationTrain[allstationID[i]]['nextBikes'])
        Xtest=np.array(allstationTest[allstationID[i]][featureCol])#
        Ytest=np.array(allstationTest[allstationID[i]]['nextBikes'])
        Ytrain[Ytrain<0.2]=-1 #classify empty station vs non empty station
        Ytrain[Ytrain>=0.2]=1
        Ytest[Ytest<0.2]=-1
        Ytest[Ytest>=0.2]=1
        
        #SVM with Gaussian Kernel
        clfSVM = ClassificationModel(SVC(), param_grid_SVM)
        clfSVM=clfSVM.fit(Xtrain, Ytrain)
        clfSVM=clfSVM.predict(Xtest,Ytest)
        SVM_model.append(clfSVM)
        ETrain_SVM[i],ETest_SVM[i]=clfSVM.getPerformanceMetric()
    
        #random forest
        clfRDforest = ClassificationModel(RandomForestClassifier(), param_grid_rd)
        clfRDforest=clfRDforest.fit(Xtrain, Ytrain)
        clfRDforest=clfRDforest.predict(Xtest,Ytest)
        SVM_model.append(clfRDforest)
        ETrain_RDforest[i],ETest_RDforest[i]=clfRDforest.getPerformanceMetric()

    return ETest_SVM,ETest_RDforest


#Adaboost models
def adaboost_mdls(allstationTrain,allstationTest,allstationID,featureCol):#adaboost-based model building
    nbstation=len(allstationID)
    
    ETrain_ada_K=np.zeros(nbstation)#training error: K represents Gaussian Kernal
    ETest_ada_K=np.zeros(nbstation)#test error
    
    ETrain_ada_L=np.zeros(nbstation)#L represents linear Kernel
    ETest_ada_L=np.zeros(nbstation)
    
    ada_K_model=[]#save all models
    ada_L_model=[]#save all models
    
    #model parameters
    nbRuns=10 #the number of boosting iterations
    
    C=1
    beta=0.002
    
    for i in range(len(allstationID)):
    #for i in range(1):
        X=np.array(allstationTrain[allstationID[i]][featureCol])#
        Y=np.array(allstationTrain[allstationID[i]]['nextBikes'])
        Xtest=np.array(allstationTest[allstationID[i]][featureCol])#
        Ytest=np.array(allstationTest[allstationID[i]]['nextBikes'])
    
        Y[Y<0.2]=-1 #classify empty station vs non empty station
        Y[Y>=0.2]=1
        Ytest[Ytest<0.2]=-1
        Ytest[Ytest>=0.2]=1
    
        Xtrain=X[:int(len(X)*0.8),:]
        Ytrain=Y[:int(len(X)*0.8)]
        Xval=X[int(len(X)*0.8):,:]
        Yval=Y[int(len(X)*0.8):]
        Ytrain[Ytrain<0.2]=-1 #classify empty station vs non empty station
        Ytrain[Ytrain>=0.2]=1
        Ytest[Ytest<0.2]=-1
        Ytest[Ytest>=0.2]=1
    
        #adaboost SVM with gaussian kernel
        ada_K = AdaboostModel(nbRuns, beta,'kernel',C)
        ada_K=ada_K.fit_predict(X, Y,Xval,Yval,Xtest,Ytest)
        ada_K_model.append(ada_K)
        ETrain_ada_K[i],valE,ETest_ada_K[i]=ada_K.getPerformanceMetric()
    
        #adaboost SVM with linear kernel
        ada_L = AdaboostModel(nbRuns, [],'linear',C)
        ada_L=ada_L.fit_predict(X, Y,Xval,Yval,Xtest,Ytest)
        ada_L_model.append(ada_L)
        ETrain_ada_L[i],valE,ETest_ada_L[i]=ada_L.getPerformanceMetric()
        
        return ETest_ada_K,ETest_ada_L


def main():
    stationstatus=pd.read_csv(folder+stationfile)
    weatherMay=pd.read_csv(folder+weatherfile_may)
    weatherJune=pd.read_csv(folder+weatherfile_june)
    weatherJuly=pd.read_csv(folder+weatherfile_july)
    weatherTotal=weatherMay.append([weatherJune,weatherJuly])
    
    allstationT=preprocess(stationstatus,weatherTotal)
    
    #get the columns of features for model building
    feature_idx1=[10]#station capacity
    featureCol=list(allstationT[list(allstationT.keys())[0]].columns.values[feature_idx1])+list(allstationT[list(allstationT.keys())[0]].columns.values[13:74])+list(allstationT[list(allstationT.keys())[0]].columns.values[75:])
    allstationID=list(allstationT.keys())#IDs of all stations
    stationLen=len(allstationT[allstationID[0]])#number of observations of each station
    nbstation=len(allstationID)#number of stations
    
    #splitting training and testing data
    ratio=0.7
    threshold=int(0.7*stationLen)
    
    #get training data for all stations
    allstationTrain,allstationTest,allpredictTrain,allpredictTest=splitdata(allstationT,threshold,allstationID)
    
    #regression model training and analysis:
    #1) Model a. ridge regression with Gaussian Kernel
    #2) Model b. ridge regression (with linear kernal)
    #3) Model c. lasso regression
    
    a,b,c=regression_mdls(allstationTrain,allstationTest,allstationID,featureCol)
    
    #classificaion model (non-Adaboost) training and analysis:
    #1) Model d: SVM with Gaussian kernel
    #2) Model e: random forest 
    
    d,e=classification_mdls(allstationTrain,allstationTest,allstationID,featureCol)
    
    #adaboost model training and analysis:
    #1) Model f: adaboost_SVM with Gaussian Kernel
    #2) Model g: adaboost_SVM with linear kernel
    f,g=adaboost_mdls(allstationTrain,allstationTest,allstationID,featureCol)
    
    #return the test errors of all the models
    return a,b,c,d,e,f,g
    
    
    


if __name__ == '__main__':

    a,b,c,d,e,f,g=main()
    




