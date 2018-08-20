# Title: product_lineup.R

#################
# Project Summary
#################

# Datasets:    
# Final Model: 
# Scenario:   Predict the sales of four different types of electronics, and 
#             evalutate the effect of service and customer reviews on sales vol-
#             ume. Recommend new lineup of products.
# Goal:        

###############
# Housekeeping
###############

rm(list = ls())
setwd("C:/Users/Litan Li/Desktop/electronidex/product_lineup")


###############
# Load packages
################

library(caret)
library(corrplot)
library(readr)


######################
# Parallel processing
######################

library(doParallel) 

# Check number of cores and workers available 
detectCores()
cl <- makeCluster(detectCores()-1, type='PSOCK')
# start parallel cluster
registerDoParallel(cl)


##############
# Import data
##############

# training/validation/test set
ExistingProducts <- read.csv("existingproductattributes2017.csv", stringsAsFactors = FALSE, header=T)

# prediction/new data
NewProducts <- read.csv("newproductattributes2017.csv", stringsAsFactors = FALSE, header=T)


################
# Evaluate data
################

#--- training/validation/test set ---#
str(ExistingProducts)  # 80 obs. of  18 variables 
summary(ExistingProducts)

# plots
barplot(table(ExistingProducts$ProductType))
hist(ExistingProducts$Price)
hist(ExistingProducts$x5StarReviews)
hist(ExistingProducts$x4StarReviews)
hist(ExistingProducts$x3StarReviews)
hist(ExistingProducts$x2StarReviews)
hist(ExistingProducts$x1StarReviews)
hist(ExistingProducts$PositiveServiceReview)
hist(ExistingProducts$NegativeServiceReview)
hist(ExistingProducts$Recommendproduct)
hist(ExistingProducts$BestSellersRank)
hist(ExistingProducts$ShippingWeight)
hist(ExistingProducts$ProductDepth)
hist(ExistingProducts$ProductWidth)
hist(ExistingProducts$ProductHeight)
hist(ExistingProducts$ProfitMargin)
hist(ExistingProducts$Volume)

plot(ExistingProducts$Price, ExistingProducts$Volume)
plot(ExistingProducts$x5StarReviews, ExistingProducts$Volume) # linear
plot(ExistingProducts$x4StarReviews, ExistingProducts$Volume)
plot(ExistingProducts$x3StarReviews, ExistingProducts$Volume)
plot(ExistingProducts$x2StarReviews, ExistingProducts$Volume)
plot(ExistingProducts$x1StarReviews, ExistingProducts$Volume)
plot(ExistingProducts$PositiveServiceReview, ExistingProducts$Volume)
plot(ExistingProducts$NegativeServiceReview, ExistingProducts$Volume)
plot(ExistingProducts$Recommendproduct, ExistingProducts$Volume)
plot(ExistingProducts$BestSellersRank, ExistingProducts$Volume)
plot(ExistingProducts$ShippingWeight, ExistingProducts$Volume)
plot(ExistingProducts$ProductDepth, ExistingProducts$Volume)
plot(ExistingProducts$ProductWidth, ExistingProducts$Volume)
plot(ExistingProducts$ProductHeight, ExistingProducts$Volume)
plot(ExistingProducts$ProfitMargin, ExistingProducts$Volume)

# check for collinearity - correlation matrix
corrAll = cor(ExistingProducts[ ,c(3:11, 13:18)], use = "all.obs", method = "pearson")
corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
corrplot(corrAll, method = "circle")
print(corrAll)

# normal quantile plot. If linear, then attribute values are normally distributed.
qqnorm(CompleteResponses$salary) 
# q-q plot. If linear line on y=x, then similar distribution. If linear but not 
# on y=x, then distributions are linearly related. 
qqplot(CompleteResponses$salary, CompleteResponses$age, plot.it = TRUE) 
# check for missing values 
anyNA(ExistingProducts[ ,12])
is.na(ExistingProducts[ ,2])


#--- prediction/new data ---#
str(SurveyIncomplete)  # 5,000 obs. of  7 variables 
summary(SurveyIncomplete)

# plot
hist(ExistingProducts$x1StarReviews)
plot(ExistingProducts$x5StarReviews, ExistingProducts$Volume)
plot(ExistingProducts$x4StarReviews, ExistingProducts$Volume)
plot(ExistingProducts$x3StarReviews, ExistingProducts$Volume)
plot(ExistingProducts$Recommendproduct, ExistingProducts$Volume)
plot(ExistingProducts$PositiveServiceReview, ExistingProducts$Volume)
plot(ExistingProducts$BestSellersRank, ExistingProducts$Volume)
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

# Imputing missing values

# BestSellersRank contains missing data. There's many ways to deal with missing 
# data but here we will simply remove the attribute.
summary(ExistingProductsD)
ExistingProductsD$BestSellersRank <- NULL

# Remove ID attribute(s)
ExistingProductsD$ProductNum <- NULL
str(ExistingProductsD)

# Check for collinearity. 
corrAll = cor(ExistingProductsD, use = "all.obs", method = "pearson")
write.csv(corrAll, "correlation_matrix_ExistingProductsD.csv")
corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
corrplot(corrAll, method = "circle")

# x4StarReviews and x5StarReviews are collinear (r = 0.879). Remove 
# x4StarReviews since it has lower correlation (0.879) with volume than 
# x5StarReviews (1).
# x2StarReviews and x3StarReviews are collinear (r = 0.861). Remove 
# x2StarReviews since it has lower correlation (0.487) with volume than 
# x3StarReviews (0.763). Further, x2StarReviews has high collinearity with 
# x1StarReviews (0.952).
# NegativeServiceReview and x1StarReview are collinear (r = 0.884). Remove 
# x1StarReview since it has lower correlation (0.255) with volume than 
# NegativeServiceReview (0.309).

ExistingProductsD24v <- ExistingProductsD
ExistingProductsD24v$x4StarReviews <- NULL
ExistingProductsD24v$x2StarReviews <- NULL
ExistingProductsD24v$x1StarReviews <- NULL
str(ExistingProductsD24v)
corrD24v = cor(ExistingProductsD24v, use = "all.obs", method = "pearson")
write.csv(corrD24v, "correlation_matrix_ExistingProductsD24v.csv")


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

















## ------- LM ------- ##
# Train model, tune, predict on test set

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

set.seed(123)
lmFit1 <- train(Volume~., data=trainSet, method="lm", trControl=fitControl)
# eval variable importance
varImp(lmFit1)
#lm variable importance

#only 20 most important variables shown (out of 22)

#Overall
#x5StarReviews               1.000e+02
#ProductTypeExtendedWarranty 1.137e-14
#ProductTypeGameConsole      1.094e-14
#ProductTypeSoftware         1.058e-14
#ShippingWeight              6.281e-15
#ProductTypeAccessories      6.247e-15
#ProductHeight               5.863e-15
#NegativeServiceReview       5.396e-15
#ProfitMargin                4.714e-15
#Recommendproduct            3.756e-15
#ProductTypePrinterSupplies  3.588e-15
#ProductTypeDisplay          3.577e-15
#x3StarReviews               3.260e-15
#ProductDepth                3.218e-15
#ProductTypeNetbook          2.926e-15
#ProductTypeSmartphone       2.057e-15
#ProductTypePrinter          1.657e-15
#Price                       8.533e-16
#ProductTypePC               7.608e-16
#PositiveServiceReview       7.216e-16


lmFit1
summary(lmFit1)
#Call:
#  lm(formula = .outcome ~ ., data = dat)

#Residuals:
#  Min         1Q     Median         3Q        Max 
#-1.727e-13 -2.735e-14  1.801e-15  2.825e-14  1.333e-13 

#Coefficients: (1 not defined because of singularities)
#Estimate Std. Error    t value Pr(>|t|)    
#(Intercept)                 -1.180e-13  8.143e-14 -1.449e+00  0.15555    
#ProductTypeAccessories      -1.140e-13  5.831e-14 -1.956e+00  0.05787 .  
#ProductTypeDisplay          -6.332e-14  5.535e-14 -1.144e+00  0.25977    
#ProductTypeExtendedWarranty -3.313e-13  9.431e-14 -3.513e+00  0.00116 ** 
#  ProductTypeGameConsole      -2.821e-13  8.342e-14 -3.382e+00  0.00168 ** 
#  ProductTypeLaptop            3.459e-15  5.404e-14  6.400e-02  0.94931    
#ProductTypeNetbook           6.095e-14  6.442e-14  9.460e-01  0.35006    
#ProductTypePC                1.984e-14  6.895e-14  2.880e-01  0.77516    
#ProductTypePrinter          -4.366e-14  7.796e-14 -5.600e-01  0.57871    
#ProductTypePrinterSupplies  -9.524e-14  8.301e-14 -1.147e+00  0.25841    
#ProductTypeSmartphone       -3.664e-14  5.374e-14 -6.820e-01  0.49952    
#ProductTypeSoftware         -1.658e-13  5.067e-14 -3.272e+00  0.00227 ** 
#  ProductTypeTablet                   NA         NA         NA       NA    
#Price                       -1.711e-17  5.418e-17 -3.160e-01  0.75386    
#x5StarReviews                4.000e+00  1.316e-16  3.040e+16  < 2e-16 ***
#  x3StarReviews               -8.736e-16  8.338e-16 -1.048e+00  0.30140    
#PositiveServiceReview       -4.673e-17  1.694e-16 -2.760e-01  0.78421    
#NegativeServiceReview        2.146e-15  1.265e-15  1.697e+00  0.09791 .  
#Recommendproduct             6.318e-14  5.272e-14  1.198e+00  0.23822    
#ShippingWeight               2.839e-15  1.444e-15  1.966e+00  0.05664 .  
#ProductDepth                 2.803e-16  2.709e-16  1.035e+00  0.30738    
#ProductWidth                 1.745e-16  3.094e-15  5.600e-02  0.95534    
#ProductHeight               -3.607e-15  1.961e-15 -1.839e+00  0.07375 .  
#ProfitMargin                 4.506e-13  3.025e-13  1.490e+00  0.14459    
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Residual standard error: 5.898e-14 on 38 degrees of freedom
#Multiple R-squared:      1,	Adjusted R-squared:      1 
#F-statistic: 3.023e+32 on 22 and 38 DF,  p-value: < 2.2e-16


# The linear model is: Volume = 4*x5StarReview with R^2 = 1 and RMSE = 2.42e-13. 
# Therefore, linear model performs very well. All other coefficients were close 
# to 0, and thus omitted for simplicity. Further, only the x5StarReview 
# coefficient was very statisfically significant. However, we were not able
# to deduce any effect that product type might have on sales volume from this
# model.
# Why did this linear model perform well? We got lucky - the way the data is as
# it is. Specifically, the number of 5 star reviews correlated with sales 
# volume almost perfectly linearly. Also, linear regression makes the assumption
# that (1) the n response random variables Y_i are independently normally 
# distriuted with (2) mean beta_0 + beta_1*x_1 + ... + beta_p*x_p where 
# p = number of independent vars and (3) each Y-i has common variance sigma^2.
# I'm not sure how to check this assumption, so whether it was fulfilled or 
# not is unknown.

lmPred1 <- predict(lmFit1, testSet)
lmPred1
#performance measurement
postResample(lmPred1, testSet$Volume)
#RMSE     Rsquared          MAE 
#6.022240e-13 1.000000e+00 2.722501e-13

#plot predicted verses actual
library(ggplot2)
ggplot() + 
  geom_point(aes(x = testSet$Volume, y = lmPred1),
             colour = 'red') + # scatter plot
  geom_abline(slope = 1, intercept = 0) + # 45 degree reference line
  ggtitle('Predicted Volume vs. Observed Volume on testSet') +
  xlab('Observed Volume') +
  ylab('Predicted Volume')



## ------- SVM ------- ##
# Train model, tune, predict on test set

## Using caret ##
# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
set.seed(123)
svmFit1 <- train(Volume~., data=trainSet, method="svmLinear", trControl=fitControl, tuneLength = 10)
svmFit1
predictors(svmFit1)
# eval variable imp
varImp(svmFit1)

svmPred1 <- predict(svmFit1, testSet)
svmPred1
#performance measurement
postResample(svmPred1, testSet$Volume)
# r^2 = 0.9998, very good fit on the test set.

#plot predicted verses actual
library(ggplot2)
ggplot() + 
  geom_point(aes(x = testSet$Volume, y = svmPred1),
             colour = 'red') + # scatter plot
  geom_abline(slope = 1, intercept = 0) + # 45 degree reference line
  ggtitle('Predicted Volume vs. Observed Volume on testSet') +
  xlab('Observed Volume') +
  ylab('Predicted Volume')


## Using e1071 ##
#svmFit2 <- svm(Volume~., data = trainSet, kernel = "linear", cost = 1)
svmFit2 <- svm(Volume~., data = trainSet, kernel = "polynomial", cost = 1, degree = 2, coef0 = 3, gamma = 1/61)
#svmFit2 <- svm(Volume~., data = trainSet, kernel = "radial", cost = 20, gamma = 1/61)
svmFit2
#Call:
#  svm(formula = Volume ~ ., data = trainSet, kernel = "linear", cost = 1)


#Parameters:
#  SVM-Type:  eps-regression 
#SVM-Kernel:  linear 
#cost:  1 
#gamma:  0.04347826 
#epsilon:  0.1 


#Number of Support Vectors:  15

predictors(svmFit2)
# eval variable imp

svmPred2 <- predict(svmFit2, testSet)
svmPred2
#performance measurement
postResample(svmPred2, testSet$Volume)
#       RMSE    Rsquared         MAE 
#249.7627645   0.9998313 110.9340396 

# r^2 = 0.9998, very good fit on the test set. Same r^2 results as using caret's
# linear SVM. This makes sense, since linear SVM has no adjustment parameters
# to differ by in the background code of separate packages.

#plot predicted verses actual
library(ggplot2)
ggplot() + 
  geom_point(aes(x = testSet$Volume, y = svmPred2),
             colour = 'red') + # scatter plot
  geom_abline(slope = 1, intercept = 0) + # 45 degree reference line
  ggtitle('Predicted Volume vs. Observed Volume on testSet') +
  xlab('Observed Volume') +
  ylab('Predicted Volume')



## ------- RF ------- ##
# Train model, tune, predict on test set

## Using caret ##
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
set.seed(123)
rfFit1 <- train(Volume~., data=trainSet, method="rf", importance=T, trControl=fitControl) 
rfFit1
# r^2 = 0.9898 good fit on the training set

predictors(rfFit1)
# eval variable imp
varImp(rfFit1)

rfPred1 <- predict(rfFit1, testSet)
rfPred1
#performance measurement
postResample(rfPred1, testSet$Volume)
# Fit on the test set is quite bad (r^2 = 0.5713). Two products, product numbers
# 150 and 198's volume were not predicted with great accuracy.

#plot predicted verses actual
library(ggplot2)
ggplot() + 
  geom_point(aes(x = testSet$Volume, y = rfPred1),
             colour = 'red') + # scatter plot
  geom_abline(slope = 1, intercept = 0) + # 45 degree reference line
  ggtitle('Predicted Volume vs. Observed Volume on testSet') +
  xlab('Observed Volume') +
  ylab('Predicted Volume')


## Using randomForest ##
set.seed(123)
rfFit2 <- randomForest(Volume~., data = trainSet, mtry = 23)
rfFit2

#Call:
#  randomForest(formula = Volume ~ ., data = trainSet, mtry = 23) 
#Type of random forest: regression
#Number of trees: 500
#No. of variables tried at each split: 23

#Mean of squared residuals: 7295.932
#% Var explained: 98.08

round(importance(rfFit2),2)

rfPred2 <- predict(rfFit2, testSet)
rfPred2

#performance measurement
postResample(rfPred2, testSet$Volume)
#RMSE     Rsquared          MAE 
#2454.3915016    0.5702645  795.4767719


# Fit on the test set is quite bad (r^2 = 0.5674). Two products, product numbers
# 150 and 198's volume were not predicted with great accuracy. I tried changing 
# mtry to other values, incase mtry=23 was overfitting, but 0.5674 was the 
# highest correlation achieved on the test set (using mtry=23).

#plot predicted verses actual
library(ggplot2)
ggplot() + 
  geom_point(aes(x = testSet$Volume, y = rfPred2),
             colour = 'red') + # scatter plot
  geom_abline(slope = 1, intercept = 0) + # 45 degree reference line
  ggtitle('Predicted Volume vs. Observed Volume on testSet') +
  xlab('Observed Volume') +
  ylab('Predicted Volume')

# Note if we removed the two data points that were not predicted accurately,
# the fit is much better on the test set r^2 = 0.9818. But, we have no reason
# to justify discarding these two points.
rfPred3 <- predict(rfFit2, testSet[-c(12, 18),])
rfPred3
#performance measurement
postResample(rfPred2, testSet[-c(12,18),]$Volume)


##--- Compare metrics using resamples() ---##

CaretModelFitResults <- resamples(list(lm=lmFit1, rf=rfFit1, svm=svmFit1))
# output summary metrics for tuned models 
summary(CaretModelFitResults)

#Call:
#  summary.resamples(object = CaretModelFitResults)

#Models: lm, rf, svm 
#Number of resamples: 100 

#MAE 
#Min.      1st Qu.       Median         Mean      3rd Qu.         Max. NA's
#lm  7.105427e-16 8.636425e-14 1.377765e-13 1.780539e-13 2.217040e-13 8.924027e-13    0
#rf  3.000320e+00 1.600627e+01 3.497058e+01 4.316154e+01 6.048719e+01 1.870162e+02    0
#svm 6.959018e-02 2.928308e+01 4.092296e+01 4.912629e+01 5.892108e+01 1.905555e+02    0

#RMSE 
#            Min.      1st Qu.       Median         Mean      3rd Qu.         Max. NA's
#lm  1.588822e-15 1.159016e-13 1.830422e-13 2.427200e-13 3.116176e-13 1.291011e-12    0
#rf  3.437383e+00 2.235186e+01 6.235321e+01 7.475017e+01 1.187657e+02 2.693460e+02    0
#svm 7.520083e-02 3.619616e+01 4.683244e+01 6.531231e+01 7.252279e+01 2.776970e+02    0

#Rsquared 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#lm  1.0000000 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000    0
#rf  0.9385888 0.9880765 0.9936063 0.9912876 0.9989581 0.9999528    0
#svm 0.9591586 0.9931329 0.9964765 0.9936073 0.9987637 1.0000000    0

# Looking at lm Rsquared results across the cross validation samples, we see 
# that there isn't ANY variation at all. This suggests that the linear model
# is overfitting.


##--- Pick Best Model and Parameters ---##

lmFit1
summary(lmFit1)
lmPred1
postResample(lmPred1, testSet$Volume)

svmFit2
svmPred2
postResample(svmPred2, testSet$Volume)

rfFit2
rfPred2
postResample(rfPred2, testSet$Volume)

# All three models above perform well on the training set, but linear regression 
# model and linear SVM model both out-perform the random forest on the test set.
# The linear model has, remarkably, a r^2 = 1 fit on the test set, however
# it also is overfitting, as suggested by the fact that the R^2 was almost the
# same across all the cross validation samples. Therefore, we choose the linear
# SVM model to make predictions.

svmFit2
svmPred2
postResample(svmPred2, testSet$Volume)

#--- Save top performing model ---#

saveRDS(svmFit2, "svmFit2.rds")  
# load and name model
svmFit2 <- readRDS("svmFit2.rds")


##################
# Predict new data 
##################

finalPred <- predict(svmFit2, newdata = NewProductsD24v)
round(finalPred,2)

#      1       2       3       4       5       6       7       8       9      10      11      12      13      14      15      16      17      18 
#352.55  185.59  289.27   50.90   13.59   84.70 1096.48   94.59   34.49 1055.22 3631.82  323.15  437.05  170.45  208.35 1766.22   23.25  187.06 
#    19      20      21      22      23      24 
#100.91  164.49  431.90   82.32   35.85 5272.56 


output <- NewProducts
output$VolumePredictions <- finalPred
write.csv(output, file="C2.T3output.csv", row.names=TRUE)

# Stop cluster when you are done
stopCluster(cl) 