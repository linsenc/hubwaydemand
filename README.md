# Bikesharing System Demand Estimation Project

In this project, we aim to predict the short term bike demand at all stations of the bike sharing system of Boston, named Hubway. The goal is to facilitate bike users for rentals and returns. For instance, we would like to answer the following question: given the number of bikes distributed across all stations at the moment, can we predict how many bikes will be at the stations of interset in the next 15 minutes. We believe this project can help a bike renter to determine if the station to return has an empty slot available in 15 minutes, as well as to determine if a station has a bike available in the next 15 minutes, such as the renter can get a bike.

In this project, we formulate this problem as a demand prediction problem, where the goal is to predict the bike station demand at the next time interval. In this project, we use various machine learning techiniques to solve this problem. This readme file provides the outline of the project, including data, models, as well as a brief discription of the content of the codes. For technical details and results, please refer to the report **Hubway_project.pdf** for more information .  

## Data
We use two sources of data: 1) hubway station status data (http://hubwaydatachallenge.org/) and 2) weather data of Boston (https://www.ncdc.noaa.gov/cdo-web/search). Both datasets are public available. In this document, as an illustration, we use a subset of station status data for three months (May, June and July) of 2012 as an illustration. This subset can be found as **stationsub3.csv** in this repository. The weather data are monthly data, they are **weatherMay.csv**, **weatherJune.csv** and **weatherJuly.csv** in this repository. For more details regarding data, please refer to the report for a more detailed description.

## Model

We use a variety of models to predict demand. The models are station based. In other words, for each station, we build an independent model. The models we've built include:

1) Regression


    We would like to predict how many bikes will be at a station in the next 15 minutes. We build three models:

    Ridge regression with Gaussian Kernel


    Ridge regression


    Lasso regression

2) Classification


    We would like to predict if a bike station will be empty in the next 15 minutes, and also, if a bike station will be full in the next 15 minutes. Both problems are binary classification problems. We build several models:

    SVM with Gaussian Kernel
    
    Random forest
    
    Adaboost SVM (with linear Kernel)
    
    Adaboost SVM with Gaussian Kernel
   
    Graphical Model Chow-and-Liu Tree

Note that we build two Adaboost algorithms with two SVM models (linear, Gaussian) as the weak classifier. Also please note that we hand-code the two Adaboost algorithms: we formulate the SVM dual form as a quadratic programming problem and solve efficiently. Except for the Chow-and-Liu tree model, all other models are included in this repository, coded in python. Expect for the two Adaboost algorithms, other models are coded with the built-in regression/classification models in scikit-learn.     




## Code structure

This project is coded in two .py files: 1) **Hubway.py** and 2) **adaboost_svm.py** or **adaboost_svm_G.py**. The main file is **Hubway.py**. File **adaboost_svm.py**/**adaboost_svm_G.py** codes the quadric programming problems with scipy optimization solver/gurobi python API and is imported in **Hubway.py**. In this part, we briefly introduce the key functions/classes in the codes.


### Functions and classes

Functions written in **Hubway.py**

function **preprocess**: this function preprocesses raw data and changes them into features. It includes three steps : 1) process station status data 2) process weather data and 3) merge. In addition, feature normalization is done in this function

function **data_clean_all_ohe**: this function transforms categorical features into dummay variables 

function **splitdata**: this function splits dataset into training data and test data

class **RegressionModel**: three regression models are summarized in this class

class **ClassificationModel**: two built-in classification models (i.e., SVM and random forest) are summarized in this class

class **AdaboostModel**: two adaboost SVM models are summarized in this class

function **regression_mdls**:this function runs three regression models

function **classification_mdls**: this function runs two built-in classification models

function **adaboost_mdls**: this function runs two adaboost SVM models

function **main**: this is the main function of the project, the main steps in the main function includes: 1) data preprocessing 2) run regression 3) run built-in classfication models and 4) run adaboost models. 

**adaboost_svm.py** includes the functions for the SVM dual optimization problem for scipy optimizer, i.e., the objective function is function **func**, the jacobian of the objective function is **func_derive**, the optimization problem is coded in **quadopt**, the adaboost algorithm is coded in **AdaBoost_SVM** and the Gaussian Kernel function is coded in **GaussianKernel**. When running the adaboost algorithms, the main function calls function **AdaBoost_SVM**.

**adaboost_svm_G.py** includes the functions for the SVM dual optimization problem for Gurobi optimizer. The optimization problem is coded in function **quadopt**, the adaboost algorithm is coded in **AdaBoost_SVM** and Gaussian Kernel function is coded in **GaussianKernel**. When running the adaboost algorithms, the main function calls function **AdaBoost_SVM**.


## Final Notes
To run script **Hubway.py** and launch the project, please download the data, change variable "folder" (in file **Hubway.py**) to your local directory and you should be good to go. Also, you could choose which optimziation (scipy/Gurobi) engine to use by specifying the which py file to import. Please let me know if there are any questions.


Linsen Chong


linsen.chong@gmail.com
