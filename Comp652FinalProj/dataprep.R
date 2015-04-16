#packages to required
require(plyr)
require(sqldf)

#
dat <- read.csv("minh_optimarevised_original.csv", header = T, na.strings=c(""," ","NA", "UNK", "UNKNOWN", "ND", "#NULL!", "UK"))
vars <- c("hypothyroid", "age", "stemi", "mi", "femal", "weight", "race", "activesmoker", "asa", "antiplatelet", "antiplateletintravenous", "baselinecreat", "peakcreatinine", "baselinehemoglobin", "nadirhemoglobin", "baselineplatelets", "baselineldl", "baselinehdl", "typeoftrop", "troppeak", "ejectionfraction", "cath", "diseasedarteries", "angioplasty", "number.ofstents", "INHOSPCABG", "shock", "stroke", "mechanicalventilation", "chf", "cardiogenicshock", "ventriculararrythmia", "atrialfibrillation", "bradyarrythmia", "arrrythmia", "cardiacarrest", "timibleed", "gibleed", "infection", "death", "diabetes", "hypertension", "priorpci", "priorrevasc", "priorcabg", "priorcvatia", "antiplateletondischarge", "bbprescribed", "statinsprescribed", "aceinhibitorprescribed", "arbprescribed", "cablocker")

#prints the names
names(dat)

#makes sure all the names match up
for(v in vars){
  print(v)
  dat[v]
}


#STEP 1 - keep required variables; correct spelling of hospital
red.dat <- dat[vars]
#names(red.dat)[names(red.dat) == "hopsital"] <- "hospital"
#hospital already dropped...


write.csv(red.dat, file="output1.csv")

#STEP 2 - turn yes no into 1, 0

red.dat.recod1 <- data.frame(apply(red.dat, 2, function(x) {x[x == "YES"] <- 1; x}))
red.dat.recod2 <- data.frame(apply(red.dat.recod1, 2, function(x) {x[x == "NO"] <- 0; x}))

#STEP 3 - turn diseased arteries into 1, 0
red.dat.recod2$diseasedarteries01 <- 0
red.dat.recod2$diseasedarteries01[!is.na(red.dat.recod2$diseasedarteries)] <- 1
red.dat.recod3 <- subset(red.dat.recod2, select = -c(diseasedarteries))

write.csv(red.dat.recod3, file="output2.csv")

#STEP 3A - turn typeoftrop into 1,0 (T=1, I=0)
red.dat.recod3$typeoftrop <- revalue(red.dat.recod3$typeoftrop, c("T"=1, 
                                                         "I"=0))

#STEP 4 - turn factors into indicators
red.dat.race.ind <- data.frame(model.matrix(~race -1, data=red.dat.recod3))
red.dat.race.ind$row.names.1 <- row.names(red.dat.race.ind)

red.dat.recod3$row.names.1 <- 1:nrow(red.dat.recod3)


red.dat.ind <- merge(x = red.dat.recod3, y = red.dat.race.ind, by = "row.names.1", all.x=TRUE)
red.dat.ind <- subset(red.dat.ind , select = -c(row.names.1, race))

#STEP 5 - recode NA's to -1 
dat.final <- data.frame(apply(red.dat.ind, 2, function(x) {x[is.na(x)] <- -1; x}))

#STEP 6 - recode all other characters into NA's
dat.final <- data.frame(apply(dat.final, 2, function(x) {x[is.character(x)] <- -1; x}))

write.csv(dat.final, file="outputFinal.csv")















