# Title: brand_preference

###############
# Project Summary
###############

# Datasets:    CompleteResponses.csv, SurveyIncomplete.csv (prediction set/new 
#              data), results.csv (results). See survey key for description of 
#              features and label.
# Final Model: rangerFit.rds (random forest). See brand_preference_tuning.xlsx
#              for a record of the performance of different models attempted and
#              of the effects of their tuning.
# Scenario:    Electronidex is an online and brick-and-mortar electronics retai-
#              ler. In order to help us decide which company to pursue a deeper 
#              strategic relationship with (between Sony or Acer), we hired a 
#              marketing research firm to conduct a survey on some of our custo-
#              mers to see which of the two brands they prefer. However, a third 
#              of the responses have missing labels (5000 examples in SurveyInc-
#              omplete.csv). We would like to predict these missing labels so we 
#              can have a more accurate suggestion of which company to pursue.
# Goal:        Predict the missing labels in SurveyInComplete.csv


###############
# Housekeeping
###############

rm(list = ls())
setwd("C:/Users/Litan/Desktop/electronidex/brand_preference")


###############
# Load packages
################

library(caret)
library(corrplot)
library(mlbench)
library(readr)


######################
# Parallel processing
######################

library(doParallel) 

# Check number of cores and workers available 
detectCores()
cl <- makeCluster(detectCores()-1, type='PSOCK')
registerDoParallel(cl)
# to stop cluster/parallel:
#stopCluster(cl) 
# to reactivate parallel:
#registerDoParallel(cl)


##############
# Import data
##############

# training/validation/test set
CompleteResponses <- read.csv("CompleteResponses.csv", stringsAsFactors = FALSE, header=T)

# prediction/new data
SurveyIncomplete <- read.csv("SurveyIncomplete.csv", stringsAsFactors = FALSE, header=T)


################
# Evaluate data
################

#--- training/validation/test set ---#
str(CompleteResponses)  # 10,000 obs. of  7 variables 
summary(CompleteResponses)

# plots
hist(CompleteResponses$salary)
hist(CompleteResponses$age)
hist(CompleteResponses$elevel)
hist(CompleteResponses$car)
hist(CompleteResponses$zipcode)
hist(CompleteResponses$credit)
hist(CompleteResponses$brand)

# check for collinearity - correlation matrix
corrAll = cor(CompleteResponses, use = "all.obs", method = "pearson")
corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
corrplot(corrAll, method = "circle")
# Does not seem to be any linearly related variables.

# normal quantile plot. If linear, then attribute values are normally distributed.
qqnorm(CompleteResponses$salary) 
# q-q plot. If linear line on y=x, then similar distribution. If linear but not 
# on y=x, then distributions are linearly related. 
qqplot(CompleteResponses$salary, CompleteResponses$age, plot.it = TRUE) 
# check for missing values 
anyNA(CompleteResponses)
is.na(CompleteResponses)


#--- prediction/new data ---#
str(SurveyIncomplete)  # 5,000 obs. of  7 variables 
summary(SurveyIncomplete)

# plot
hist(SurveyIncomplete$salary)
hist(SurveyIncomplete$age)
hist(SurveyIncomplete$elevel)
hist(SurveyIncomplete$car)
hist(SurveyIncomplete$zipcode)
hist(SurveyIncomplete$credit)

qqnorm(SurveyIncomplete$credit) #normal quantile plot. 
qqplot(SurveyIncomplete$salary, SurveyIncomplete$age, plot.it = TRUE) # q-q plot. 
# check for missing values 
anyNA(SurveyIncomplete)
is.na(SurveyIncomplete)


##########################################
# Preprocess Data and Feature Engineering
##########################################


# Encode all categorical variables to numeric data as dummy
# variables. * NOT PERFORMED* since random forest can handle categorical data.

# Change categorical variables to factor/ordered. See http://appliedpredictivemodeling.com/blog/2013/10/23/the-basics-of-encoding-categorical-data-for-predictive-models
# for a discussion on encoding to factor vs. encoding to ordinal. 
# Change elevel to ordered. Treat other categorical variables as nominal. 

# Note that caret dummy() (we're not using it here for random forest since 
# random forest can handle categorical features) turns ordered catergorical 
# variables into columns of (standardized) values where the value varies 
# linearly, quadraticly, cubicly (and so on) with the levels. So if the 
# underlying relationship between the ordered categorical feature and the label 
# is for example cubic, we have a feature of data that captures that trend. 
# See link above for more details.

# Change variable types 
str(CompleteResponses)
CompleteResponses$elevel  <- as.ordered(CompleteResponses$elevel)
CompleteResponses$car     <- as.factor(CompleteResponses$car)
CompleteResponses$zipcode <- as.factor(CompleteResponses$zipcode)
CompleteResponses$brand   <- as.factor(CompleteResponses$brand)

str(SurveyIncomplete)
SurveyIncomplete$elevel  <- as.ordered(SurveyIncomplete$elevel)
SurveyIncomplete$car     <- as.factor(SurveyIncomplete$car)
SurveyIncomplete$zipcode <- as.factor(SurveyIncomplete$zipcode)

# remove label column from prediction/new data - we'll predict this
SurveyIncomplete$brand <- NULL

str(CompleteResponses)
str(SurveyIncomplete)

# Create dummy variables. No linear dependencies: fullRank = TRUE
# * NOT PERFORMED* since random forest can handle categorical data.
#dummies1 <- dummyVars(~., CompleteResponses, fullRank = TRUE)
#CompleteResponsesDV <- data.frame(predict(dummies1, CompleteResponses))
#names(CompleteResponsesDV)[names(CompleteResponsesDV)=="brand.1"] <- "brand"
#CompleteResponsesDV$brand <- as.factor(CompleteResponsesDV$brand)
#str(CompleteResponsesDV)

#dummies2 <- dummyVars(~., SurveyIncomplete, fullRank = TRUE)
#SurveyIncompleteDV <- data.frame(predict(dummies2, SurveyIncomplete))
#str(SurveyIncompleteDV)

# make polynomial features
#* NOT PERFORMED* since it turns out our random forest model is not underfitting 
# with just the original set of features.

# Split into training/val set and test set. We will use k-fold cross validation
# when training, so training and validation examples will both be pulled from 
# trainSet 
inTraining <- createDataPartition(CompleteResponses$brand, p=0.8, list=FALSE)
trainSet <- CompleteResponses[inTraining,]   
testSet <- CompleteResponses[-inTraining,]   

# scale features
scaleParamsTrain <- preProcess(trainSet[, c(1,2,6)], 
                               method = c("center", "scale"))
print(scaleParamsTrain) 

trainSetS <- predict(scaleParamsTrain, trainSet)
testSetS <- predict(scaleParamsTrain, testSet) # scaled with training set means and std. devs.
predictionSetS <- predict(scaleParamsTrain, SurveyIncomplete) # scaled with train set means and std. devs.

# save data
write.csv(trainSetS, "rf_trainSetS.csv", row.names = FALSE)
write.csv(testSetS, "rf_testSetS.csv", row.names = FALSE)
write.csv(predictionSetS, "rf_predictionSetS.csv", row.names = FALSE)
#trainSetS      <- read.csv("rf_trainSetS.csv")
#testSetS       <- read.csv("rf_testSetS.csv")
#predictionSetS <- read.csv("rf_predictionSetS.csv")


#################
# Train model(s)
#################

## ------- Random Forest ------- ##

# 10-fold cross validation. 
# summaryFunction: classification -> Data not skewed -> Use accuracy and kappa 
rangerFitControl <- trainControl(method = "cv", number = 10,
                             summaryFunction = defaultSummary)

# hyperparameter values to try
modelLookup("ranger")
rangerGrid <- expand.grid(mtry = c(2,4,6,8,10),
                          splitrule = c("gini"),
                          min.node.size = c(7,9))
nrow(rangerGrid)

# Try different sets of hyperparameters, pick the best set based on best 
# cross-validated accuracy (and kappa), and fit the final model to all the 
# training data using the optimal hyperparameter set.
startTime <- Sys.time()
#rfFit <- train(x = trainSetS[ ,1:6], y = trainSetS$brand, 
#               method="rf", 
#               trControl=rfFitControl,
#               tuneLength = 5) 
rangerFit <- train(x = trainSetS[ ,1:6], y = trainSetS$brand,
                   method = "ranger",
                   trControl = rfFitControl,
                   tuneGrid = rangerGrid)
endTime <- Sys.time()
rangerFitRunTime <- endTime - startTime


print(rangerFit) 
# cross-validated accuracy and kappa - accuracy 0.921, kappa 0.833
confusionMatrix(rangerFit, norm = "none") # confusion matrix for the hold-out samples
ggplot(rangerFit)
ggplot(rangerFit, metric = "Kappa")
varImp(rangerFit) # most important features


print(rangerFit$finalModel)  
# OOB accuracy  - OOB accuracy 1-0.079 = 0.921 


trainPred <- predict(rangerFit, trainSetS[ ,1:6])
confusionMatrix(data = trainPred, reference = trainSetS$brand) 
# trainingset accuracy and kappa - accuracy 0.968, kappa 0.931


testPred <- predict(rangerFit, testSetS[ ,1:6])
confusionMatrix(data = testPred, reference = testSetS$brand) 
# testset accuracy and kappa - accuracy 0.970, kappa 0.936

# save final model
saveRDS(rangerFit, "rangerFit.rds") 
#rangerFit <- readRDS("rangerFit.rds")


##################
# Predict new data
##################

# predict with best model
predictionPred <- predict(rangerFit, predictionSetS)
SurveyIncomplete$brand <- predictionPred
write.csv(SurveyIncomplete, "results.csv")
summary(predictionPred)
# In the incomplete survey, we predict that 1917 of those surveyed would have 
# prefered Acer while 3083 would have prefered Sony.


#--- Conclusion ---#

# Top performing model: random forest (ranger) with mtry = 4, min.node.size = 9, 
# splitrule = "gini". 

# Reason: 
# -Accuracy and kappa on the training set isn't 1, which suggests 
# it isn't overfitting as the previous models were. 
# -Good performance on the test set - generalizes well.

# cross-validated accuracy and kappa - accuracy 0.921, kappa 0.833
# trainingset accuracy and kappa - accuracy 0.968, kappa 0.931
# testset accuracy and kappa - accuracy 0.970, kappa 0.936


# stop cluster when done
stopCluster(cl)