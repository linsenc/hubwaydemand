#random forest
setwd("C:/MIT courses/IAP/ORSoftwareTools2014-master/IntermediateR")
bikeTrain = read.csv("bikeTrain.csv", header=FALSE,stringsAsFactors=TRUE)
bikeVal=read.csv("bikeVal.csv", header=FALSE,stringsAsFactors=TRUE)
bikeTest=read.csv("bikeTest.csv", header=FALSE,stringsAsFactors=TRUE)

bikeTrain$V77=factor(bikeTrain$V77)
bikeVal$V77=factor(bikeVal$V77)
bikeTest$V77=factor(bikeTest$V77)


#bikeForest=randomForest(V77 ~ .,data=bikeTrain,nodesize = 10, ntree = 200)

#bikePred = predict(bikeForest, newdata = bikeVal)
#bike.table=table(bikeVal$V77, bikePred)
#bike.table

# Check accuracy
##sum(diag(bike.table))/nrow(bikeVal)


#nodesize  
#Minimum size of terminal nodes. Setting this number larger causes smaller trees to be grown (and thus take less time). Note that the default values are different for classification (1) and regression (5).


nbtree=seq(50,400,by=50)
nsize=seq(2,20,by=2)

misclassVal=array(0,dim=c(length(nbtree),length(nsize)))
misclassTest=array(0,dim=c(length(nbtree),length(nsize)))


for (i in 1:length(nbtree)) 
{
  for (j in 1:length(nsize)){
    bikeForest=randomForest(V77 ~ .,data=bikeTrain,nodesize =nsize[j] , ntree = nbtree[i])
    
    bikePred = predict(bikeForest, newdata = bikeVal)
    bikePredTest=predict(bikeForest, newdata = bikeTest)
    bike.table=table(bikeVal$V77, bikePred)
    bike.test=table(bikeTest$V77, bikePredTest)
    #bike.table
    
    # Check accuracy
    misclassVal[i,j]=1-sum(diag(bike.table))/nrow(bikeVal)
    misclassTest[i,j]=1-sum(diag(bike.test))/nrow(bikeTest)
  }
}
View(misclassVal)
write.csv(misclassVal,file='rdmForest55.csv')

