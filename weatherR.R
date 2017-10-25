setwd("C:/MIT courses/IAP/ORSoftwareTools2014-master/IntermediateR")
weatherMay = read.csv("weatherMay.csv", stringsAsFactors=FALSE)
weatherJune=read.csv("weatherJune.csv", stringsAsFactors=FALSE)
weatherJuly=read.csv("weatherJuly.csv", stringsAsFactors=FALSE)
weatherTot=rbind(weatherMay,weatherJune,weatherJuly)

a=subset(weatherTot, select=c(Date,Time,Visibility,Rain,Drizzle,Mist,Fog,DryBulbFarenheit,RelativeHumidity,WindSpeed))
weatherTot=a
weatherTot$Rain=1-is.na(weatherTot$Rain)
weatherTot$Drizzle=1-is.na(weatherTot$Drizzle)
weatherTot$Mist=1-is.na(weatherTot$Mist)
weatherTot$Fog=1-is.na(weatherTot$Fog)
  
#use max to merge binary variables, use mean to merge numerical values
  #use tapply again

write.csv(weatherTot,file='weather1.csv')

weatherTot$hour=weatherTot$Date*100+floor(weatherTot$Time/100)
weather2=subset(weatherTot, ((weatherTot$Date>20120505) & (weatherTot$Date<20120730)))

#weather hour is about the weather to use for the afternoon hours
weatherHour=subset(weather2, ((floor(weather2$Time/100)>15) & (floor(weather2$Time/100<19))))
att=tapply(weatherHour$Rain,weatherHour$hour,max)
weatherHourNew=data.frame(att)

weatherHourNew$Rain=tapply(weatherHour$Rain,weatherHour$hour,max)
weatherHourNew$Drizzle=tapply(weatherHour$Drizzle,weatherHour$hour,max)
weatherHourNew$Mist=tapply(weatherHour$Mist,weatherHour$hour,max)
weatherHourNew$Fog=tapply(weatherHour$Fog,weatherHour$hour,max)

weatherHourNew$Temp=tapply(weatherHour$DryBulbFarenheit,weatherHour$hour,mean)
weatherHourNew$Humid=tapply(weatherHour$RelativeHumidity,weatherHour$hour,mean)
weatherHourNew$Visibility=tapply(as.numeric(weatherHour$Visibility),weatherHour$hour,mean)
weatherHourNew$WindSpeed=tapply(weatherHour$WindSpeed,weatherHour$hour,mean)
weatherHourNew$hourID=as.numeric(names(weatherHourNew$Rain))

write.csv(weatherHourNew,file='weather2.csv')





