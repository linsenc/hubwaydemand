### Bikesharing System Demand Estimation Project

In this project, we aim to predict the short term bike demand at all stations of the bike sharing system of Boston, named Hubway. The goal is to facilitate bike users for rentals and returns. The goal is two be able to answer the following question: given the bike distributions at the time of interest, can we predict how many bikes will be at the stations of interset in 15 minutes. We believe this project can help a bike renter to determine if the destination station has an empty slot available in 15 minutes, as well as to determine if a station has a bike in the next 15 minutes, such as a potential renter can get a bike.

In the project, we formulate this problem as a demand estimation problem, where the goal is to predict the bike station demand as the next time interval. In this project, we use various machine learning techiniques to solve this problem. This readme file provides the outline of the project, including data, models, as well as a brief discription of the content of the codes. For technical details and results, please refer to the report (Hubway_project.pdf) for more information .  

## Data
We use two sources of data: 1) hubway station status data (from http://hubwaydatachallenge.org/) and 2) weather data of Boston (https://www.ncdc.noaa.gov/cdo-web/search). Both datasets are public available. In this document, as an illustration, we use a subset of station status data for three months (May, June and July) of 2012 as an illustration. This subset can be found as "stationsub3.csv" in this repository. The weather data are monthly data, we save them as "weatherMay.csv", "weatherJune.csv" and "weatherJuly.csv". For more details regarding data, please refer to the report for a more detailed description.

## Model

We use a variety of models to predict demand. The models are station based. Thus, for each station, we build an independent model. The models we've built include 

1) Regression.

Adaboost SVM. Hand code, optimization algorithm, and it's formulation is shown in

There are two main .py files. 1) 2)

describe data

read the classes.

faciliate read the codes

In file 2), we put the functions

## Code structure
