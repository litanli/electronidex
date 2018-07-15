# Title: brand_preference

###############
# Project Notes
###############




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
# variables.

# First change categorical variables to factor/ordered. See http://appliedpredictivemodeling.com/blog/2013/10/23/the-basics-of-encoding-categorical-data-for-predictive-models
# for a discussion on encoding factor vs ordinal. 
# Change elvel to ordered. We'll attempt to capture underlying patterns between 
# elevel and brand using polynomial feature terms. Treat other categorical 
# variables as nominal.
str(CompleteResponses)
CompleteResponses$elevel  <- as.ordered(CompleteResponses$elevel)
CompleteResponses$car     <- as.factor(CompleteResponses$car)
CompleteResponses$zipcode <- as.factor(CompleteResponses$zipcode)
CompleteResponses$brand   <- as.factor(CompleteResponses$brand)

str(SurveyIncomplete)
SurveyIncomplete$elevel  <- as.ordered(SurveyIncomplete$elevel)
SurveyIncomplete$car     <- as.factor(SurveyIncomplete$car)
SurveyIncomplete$zipcode <- as.factor(SurveyIncomplete$zipcode)

# remove label column from prediction/new data - shouldn't be there
SurveyIncomplete$brand <- NULL

str(CompleteResponses)
str(SurveyIncomplete)

# Then create dummy variables. No linear dependencies: FullRank = TRUE
dummies1 <- dummyVars(~., CompleteResponses, fullRank = TRUE)
CompleteResponsesDV <- data.frame(predict(dummies1, CompleteResponses))
names(CompleteResponsesDV)[names(CompleteResponsesDV)=="brand.1"] <- "brand"
CompleteResponsesDV$brand <- as.factor(CompleteResponsesDV$brand)
str(CompleteResponsesDV)

dummies2 <- dummyVars(~., SurveyIncomplete, fullRank = TRUE)
SurveyIncompleteDV <- data.frame(predict(dummies2, SurveyIncomplete))
str(SurveyIncompleteDV)

# Split into training/val set and test set. We will use k-fold cross validation
# when training, so training and validation examples will both be pulled from 
# trainSet 
inTraining <- createDataPartition(CompleteResponsesDV$brand, p=0.8, list=FALSE)
trainSet <- CompleteResponsesDV[inTraining,]   
testSet <- CompleteResponsesDV[-inTraining,]   

# scale features
scaleParamsTrain <- preProcess(trainSet[, c(1,2,34)], 
                               method = c("center", "scale"))
print(scaleParamsTrain) 

trainSetS <- predict(scaleParamsTrain, trainSet)
testSetS <- predict(scaleParamsTrain, testSet) # scaled with trainSet params
predictionSetS <- predict(scaleParamsTrain, SurveyIncompleteDV) # scaled with trainSet params


#################
# Train model(s)
#################

## ------- Regularized Logistic Regression ------- ##

# 10-fold cross validation. 
# summaryFunction: classification -> Data not skewed -> Use accuracy and kappa 
regLogisticFitControl <- trainControl(method = "cv", number = 10,
                            summaryFunction = defaultSummary)

# hyperparameter values to try
modelLookup("regLogistic")
regLogisticGrid <- expand.grid(cost = c(0,0.03,0.1,0.3,1,3,10), 
                    loss = c("L1", "L2_dual", "L2_primal"),
                    epsilon = c(0.01))
nrow(regLogisticGrid)

startTime <- Sys.time()
regLogisticFit2 <- train(x = CompleteResponses[ ,1:6], y = CompleteResponses[ ,7], 
                        method="regLogistic", 
                        trControl=regLogisticFitControl) 
endTime <- Sys.time()
regLogisticFit2RunTime <- endTime - startTime

print(regLogisticFit2) 
plot(regLogisticFit2)

saveRDS(regLogisticFit2, "regLogisticFit2.rds") 
## Model performance on validation set (cross-validation accuracy and kappa)
#Regularized Logistic Regression 

#10000 samples
#6 predictor
#2 classes: '0', '1' 

#Resampling: Cross-Validated (10 fold) 
#Summary of sample sizes: 9000, 9000, 8999, 9000, 9000, 9000, ... 
#Resampling results across tuning parameters:
  
#  cost  loss       epsilon  Accuracy   Kappa        
#0.5   L1         0.001    0.5378026  -0.1173015808
#0.5   L1         0.010    0.5575045  -0.0846897333
#0.5   L1         0.100    0.6031016  -0.0296136429
#0.5   L2_dual    0.001    0.5298073  -0.0127066432
#0.5   L2_dual    0.010    0.5510156  -0.0271783953
#0.5   L2_dual    0.100    0.5331036   0.0049793299
#0.5   L2_primal  0.001    0.6048001  -0.0298731287
#0.5   L2_primal  0.010    0.6048001  -0.0298731287
#0.5   L2_primal  0.100    0.6048001  -0.0298731287
#1.0   L1         0.001    0.5379022  -0.1162816303
#1.0   L1         0.010    0.5498035  -0.0976935905
#1.0   L1         0.100    0.5874963  -0.0527830202
#1.0   L2_dual    0.001    0.5288000   0.0133878554
#1.0   L2_dual    0.010    0.5556055   0.0013410521
#1.0   L2_dual    0.100    0.5285900   0.0115473719
#1.0   L2_primal  0.001    0.6048001  -0.0298731287
#1.0   L2_primal  0.010    0.6048001  -0.0298731287
#1.0   L2_primal  0.100    0.6048001  -0.0298731287
#2.0   L1         0.001    0.5373039  -0.1174711289
#2.0   L1         0.010    0.5564047  -0.0862131117
#2.0   L1         0.100    0.5922001  -0.0481335396
#2.0   L2_dual    0.001    0.5581909  -0.0003878388
#2.0   L2_dual    0.010    0.5594243  -0.0138462892
#2.0   L2_dual    0.100    0.5567757  -0.0177376307
#2.0   L2_primal  0.001    0.6048001  -0.0298731287
#2.0   L2_primal  0.010    0.6048001  -0.0298731287
#2.0   L2_primal  0.100    0.6048001  -0.0298731287

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were cost = 0.5, loss = L2_primal
#and epsilon = 0.001.


#C5.0 

#7501 samples
#6 predictor
#2 classes: '0', '1' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 6751, 6750, 6751, 6751, 6751, 6750, ... 
#Resampling results across tuning parameters:
  
#  model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.8785408  0.7498139
#rules  FALSE   10      0.9194734  0.8279742
#rules   TRUE    1      0.8906743  0.7727375
#rules   TRUE   10      0.9194736  0.8279380
#tree   FALSE    1      0.8754734  0.7383688
#tree   FALSE   10      0.9185406  0.8270559
#tree    TRUE    1      0.8912076  0.7737408
#tree    TRUE   10      0.9180076  0.8258600

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 10, model = rules and winnow 
# = TRUE.

# Note that the trials = 10, model = rules and winnow = FALSE model had 
# comparable accuracy and ever so slightly higher kappa; but, caret used the 
# model with the highest accuracy.

# eval variable imp. For C5.0, predictor importance is measured by determining 
# the percentage of training set examples that fall into all the terminal nodes 
# after the split by the predictor.
varImp(C5.0_Fit)

#C5.0 variable importance

#only 20 most important variables shown (out of 34)

#Overall
#age          100
#salary       100
#car17          0
#car19          0
#car14          0
#car16          0
#zipcode7       0
#zipcode2       0
#car4           0
#car11          0
#car18          0
#elevel.C       0
#zipcode6       0
#zipcode1       0
#car7           0
#zipcode3       0
#car9           0
#car10          0
#zipcode4       0
#elevel.L       0

# Based on the C5.0 tree, salary and age were by far the best predictors for 
# brand preference.


## ------- Random Forest ------- ##

# train/fit
set.seed(123)
# stopCluster(cl); print("Cluster stopped."); registerDoSEQ() # required for 
# train() to work manually specify tuning parameters to try. mtry is the number 
# of variables randomly sampled as candidates at each split. 
# Default value for classification = sqrt(p) where p = number of variables in 
# dataset.
grid <- expand.grid(mtry=c(2,3,4,5,6))
# "tuneLength = 5" - automatically tune parameters, using 2 different values per 
# parameter for total of 4 parameter combinations
rf_Fit <- train(brand~., data=trainSet, method="rf", trControl=fitControl, tuneGrid = grid) 
rf_Fit # training time ~ 3 minutes

#Random Forest 

#7501 samples
#6 predictor
#2 classes: '0', '1' 

#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 1 times) 
#Summary of sample sizes: 6751, 6750, 6751, 6751, 6751, 6750, ...,  
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2     0.6216505  0.0000000
#3     0.7375058  0.3754540
#4     0.8377551  0.6467993
#5     0.8825462  0.7503174
#6     0.9000097  0.7882906

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 6.

# eval variable imp
varImp(rf_Fit)

#rf variable importance

#only 20 most important variables shown (out of 34)

#Overall
#salary   100.0000
#age       45.9201
#credit    20.4728
#elevel.L   1.7031
#elevel.C   1.6522
#elevel^4   1.6265
#elevel.Q   1.2213
#zipcode5   0.6925
#zipcode3   0.6871
#zipcode4   0.6533
#zipcode1   0.6118
#zipcode6   0.6089
#zipcode7   0.5403
#zipcode2   0.4202
#zipcode8   0.4073
#car7       0.3823
#car5       0.3147
#car15      0.3146
#car2       0.2675
#car12      0.2609

# based on the RF model, salary, age, and credit were the most important 
# predictors.


#--- Conclusion ---#

# Training set: training set partition of CompleteResponses
# Top performing model: C5.0 tree with model = rules  winnow = TRUE trials = 10. 
# Accuracy = 0.9194736 kappa = 0.8279380
# varImp: Based on the C5.0 tree, salary and age were by far the best predictors 
# for brand preference.



#--- Save top performing model ---#

saveRDS(C5.0_Fit, "C5.0_Fit.rds")  
# load and name model
C5.0_Fit <- readRDS("C5.0_Fit.rds")



#################
# Predict testSet
#################
C5.0_Pred1 <- predict(C5.0_Fit, testSet)
C5.0_Pred1
# performance measurement
postResample(C5.0_Pred1, testSet$brand)
#  Accuracy     Kappa 
# 0.9223689 0.8338324 
# Pretty good!

# plot predicted verses observed
plot(C5.0_Pred1,testSet$brand)



##################
# Predict new data
##################

# predict with C5.0
C5.0_Pred2 <- predict(C5.0_Fit, SurveyIncomplete_N)
C5.0_Pred2
writeClipboard(as.character(C5.0_Pred2))
summary(C5.0_Pred2)
# In the incomplete survey, we predict that 1893 of those surveyed would have 
# prefered Acer while 3107 would have prefered Sony.

# Again, based on our optimal C5.0 model, we found that salary and age were the 
# best predictors for brand preference, with each assigned a variable important 
# value of 100 by the varImp() function.



# Summarize project (place at top): We will train a model to predict customer brand preferences 
# using CompleteResponses.csv as the training set, picking the best classifier 
# out of a couple of types of decision trees. We will then test the best model 
# using SurveyIncomplete.csv as the test set.


# stop cluster when done
stopCluster(cl)