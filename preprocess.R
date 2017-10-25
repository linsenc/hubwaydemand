setwd("C:/MIT courses/IAP/ORSoftwareTools2014-master/IntermediateR")
stationstatus = read.csv("stationsub3.csv", stringsAsFactors=FALSE)

st_may=subset(stationstatus,stationstatus$month==5)
st_june=subset(stationstatus,stationstatus$month==6)
st_july=subset(stationstatus,stationstatus$month==7)
st_july_sub=subset(st_july,st_july$day<30)
st_may_sub=subset(st_may,st_may$day>5)
st4=rbind(st_may_sub,st_june,st_july_sub)
# now first split by station, then mean with 15 minutes, then figure out other features
#spl=split(st4,st4$station_id)

summary(spl[[0]])
summary(spl[[2]])
#stat2
quarter1=floor(st4$time$min/15)+1
#figure out the mean

st4=stationstatus

st4$quarter=floor(st4$time$min/15)+1
write.csv(st4,file='stationsub3.csv')
st4$time_station_ID=(st4$month*100000+st4$day*1000+st4$time$hour*10+st4$quarter)*100+st4$station_id
st4$time = strptime(st4$time, "%Y-%m-%d %H:%M:%S")

st4$wday=st4$time$wday

#spl1=split(st4,st4$time_station_ID)
#spl2=lapply(spl1,mean)
st5=data.frame()

at=tapply(st4$wday,st4$time_station_ID,mean)
st5=data.frame(at)
st5$wday=tapply(st4$wday,st4$time_station_ID,mean)
st5$nbBikes=tapply(st4$nbBikes,st4$time_station_ID,mean)
st5$nbEmptyDocks=tapply(st4$nbEmptyDocks,st4$time_station_ID,mean)
st5$capacity=tapply(st4$capacity,st4$time_station_ID,mean)
st5$time_station_ID=as.numeric(names(st5$nbBikes))
#stAve(time_station_D,nbBikes,nbEmptyDocks,capacity)
st5=data.frame(stAve)
write.csv(st5,file='stationsub4.csv')
