####Author: Ojas Mehta
library(MASS)
library(caret)
library(caTools)
library(ggplot2)
library(lattice)
library(corrplot)
library(magrittr) # need to run every time you start R and want to use %>%
library(dplyr)
library(tidyverse) # for tidy data analysis
library(readr)     # for fast reading of input files
library(mice)      # mice package for Multivariate Imputation by Chained Equations (MICE)
###for models
library(rpart)
library(randomForest)
library(e1071)#svm
library(C50)#decisiontree
library(elmNN)

dbdata <- read.csv("Data/diabetes.csv", header = TRUE)
set.seed(1234)
###check for missing values####
print(table(is.na(dbdata)))
##ans: FALSE 6912 so no missing value was there
#
dbdata$Outcome =as.factor(dbdata$Outcome) #if classification else it will do regression
trainIndex = createDataPartition(dbdata$Outcome, p=0.8,list=FALSE)

dbtrain = dbdata[trainIndex,]
dbtest = dbdata[-trainIndex,]

train.X<-subset(dbtrain,select=c(-Outcome))
train.y<-subset(dbtrain,select=c(Outcome))

test.X<-subset(dbtest,select=c(-Outcome))
test.y<-subset(dbtest,select=c(Outcome))


####Scaling data#########
stdData = preProcess(train.X, method = c("center", "scale"))
train.X <- predict(stdData,train.X)
test.X <- predict(stdData,test.X)


#univariate visualization
c2 <- rainbow(8, alpha=0.2)
for(i in 1:8) {
  boxplot(train.X[,i], main=names(train.X)[i], col = c2[i])
}


# check that we get mean of 0 and sd of 1
print(colMeans(train.X))
print(apply(train.X, 2, sd))

###Multivariate visualization of data#####3
c1 <- rainbow(8)
c2 <- rainbow(8, alpha=0.2)
c3 <- rainbow(8, v=0.7)
boxplot(train.X, main = "Train set (after standardization)", col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2)

dbtrain$Outcome<-as.numeric(as.character(dbtrain$Outcome))
dbtest$Outcome<-as.numeric(as.character(dbtest$Outcome))
corrplot(cor(dbtrain))
###train test split features distribution
bind_rows(data.frame(group = "train", dbtrain),
          data.frame(group = "test", dbtest)) %>%
  gather(x, y, Pregnancies:Age) %>%
  ggplot(aes(x = y, color = group, fill = group)) +
  geom_density(alpha = 0.3) +
  facet_wrap( ~ x, scales = "free", ncol = 3)



#####For Classification
dbtrain$Outcome<-as.factor(dbtrain$Outcome)
dbtest$Outcome<-as.factor(dbtest$Outcome)


###different models
trcontrol = trainControl(method = "cv",number=10,savePredictions = TRUE,verboseIter = FALSE)


set.seed(1234)
#####parallel randomForest
tunegrid = expand.grid(mtry = c(2:8))
Model1 = train(Outcome ~ ., data = dbtrain, method = "parRF", trainControl = trcontrol, tuneGrid = tunegrid)
ModelPred1 = predict(Model1, dbtest)

set.seed(1234)
##### randomForest
tunegrid = expand.grid(mtry = c(2:8))
Model2 = train(Outcome ~ ., data = dbtrain, method = "rf", trainControl = trcontrol, tuneGrid = tunegrid)
ModelPred2 = predict(Model2, dbtest)

set.seed(1234)
#########svm with gaussian kernel
tunegrid = expand.grid(sigma = c(0.01, 0.014, 0.015, 0.017), C = c(1, 2, 3))
Model3 = train(Outcome ~ ., data = dbtrain, method = "svmRadial", trainControl = trcontrol, tuneGrid = tunegrid)
ModelPred3 = predict(Model3, dbtest)

set.seed(1234)
#########c5.0 decision tree
tunegrid = expand.grid( winnow = c(TRUE,FALSE), trials=c(17, 19, 20, 22, 24), model="tree" )
Model4 = train(Outcome ~ ., data = dbtrain, method = "C5.0", trainControl = trcontrol, tuneGrid = tunegrid)
ModelPred4 = predict(Model4, dbtest)


set.seed(1234)
# #########extreme learning machine with gaussian kernel
tunegrid = expand.grid( nhid = c(15, 20, 30, 40, 50, 60), actfun = c("radbas", "sig", "sin", "poslin","tansig", "purelin"))
Model5 = train(Outcome ~ ., data = dbtrain, method = 'elm', trainControl = trcontrol, tuneGrid = tunegrid)
ModelPred5 = predict(Model5, dbtest)

set.seed(1234)
#########avNNet
tunegrid = expand.grid(size = c(1, 2, 3, 4, 5), decay = c(0.5, 1, 2), bag=FALSE)
Model6 = train(Outcome ~ ., data = dbtrain, method = 'avNNet', trainControl = trcontrol, tuneGrid = tunegrid)

ModelPred6 = predict(Model6, dbtest)


##########Replace model1 by the model whose results are required
print(summary(Model1))
print(confusionMatrix(table(dbtest$Outcome, ModelPred1)))
ggplot(Model1)