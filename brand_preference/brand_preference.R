###############
# Project Summary
###############

# Datasets:    complete_responses.csv (training/validation/test set), 
#              survey_incomplete.csv (prediction set/new data), predictions.csv 
#              (predictions made on survey_incomplete.csv). See survey_key.xlsx 
#              for a description of the features and the label.
#
# Final Model: ranger_fit.rds (random forest). See "tuning.xlsx" for a record of 
#              the performance of different models attempted and of the effects 
#              of their hyperparameter tuning.
#
# Scenario:    Electronidex is a mock online and brick-and-mortar electronics 
#              retailer. In order to help us decide which company to pursue a deeper 
#              strategic relationship with (between Sony or Acer), we hired a 
#              marketing research firm to conduct a survey on some of our custo-
#              mers to see which of the two brands they prefer. However, a third 
#              of the responses have missing labels (5000 examples in 
#              survey_incomplete.csv). We would like to predict these missing labels so we 
#              can have a more accurate suggestion of which company to pursue.

# Goal:        Predict the missing labels in survey_incomplete.csv
#
# Conclusion:  Top performing model: random forest (ranger) with mtry = 4, 
#              min.node.size = 9, splitrule = "gini". 
#              cross-validated accuracy and kappa - accuracy 0.921, kappa 0.833
#              training_set accuracy and kappa - accuracy 0.970, kappa 0.933
#              test_set accuracy and kappa - accuracy 0.973, kappa 0.942
#              In the incomplete survey, we predict that 1915 of those surveyed 
#              would have prefered Acer while 3085 would have prefered Sony.
# 
# Reason:      Accuracy and kappa on the test set are high. 
#              Good performance on the test set - generalizes well.
#              Gap between the cross-validation accuracy and the training set
#              accuracy is small, suggesting the model is not overfitting too
#              much. Admittedly, the gap in kappa is bigger.

              
###############
# Housekeeping
###############

rm(list = ls())
setwd("C:/Users/Litan Li/Desktop/electronidex/brand_preference")

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
# Start parallel cluster
registerDoParallel(cl)


##############
# Import data
##############

# training/validation/test set
complete_responses <- read.csv("complete_responses.csv", 
                               stringsAsFactors = FALSE, header=T)

# prediction/new data
survey_incomplete <- read.csv("survey_incomplete.csv", 
                              stringsAsFactors = FALSE, header=T)


################
# Evaluate data
################

#--- training/validation/test set ---#
str(complete_responses)  # 10,000 obs. of  7 variables 
summary(complete_responses)

# plots
hist(complete_responses$salary)
hist(complete_responses$age)
hist(complete_responses$elevel)
hist(complete_responses$car)
hist(complete_responses$zipcode)
hist(complete_responses$credit)
hist(complete_responses$brand)

# check for collinearity - correlation matrix
corrAll = cor(complete_responses, use = "all.obs", method = "pearson")
corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
corrplot(corrAll, method = "circle")
# Does not seem to be any linearly related variables.

# normal quantile plot. If linear, then attribute values are normally distributed.
qqnorm(complete_responses$salary) 
# q-q plot. If linear line on y=x, then similar distribution. If linear but not 
# on y=x, then distributions are linearly related. 
qqplot(complete_responses$salary, complete_responses$age, plot.it = TRUE) 
# check for missing values 
anyNA(complete_responses)
is.na(complete_responses)


#--- prediction/new data ---#
str(survey_incomplete)  # 5,000 obs. of  7 variables 
summary(survey_incomplete)

# plot
hist(survey_incomplete$salary)
hist(survey_incomplete$age)
hist(survey_incomplete$elevel)
hist(survey_incomplete$car)
hist(survey_incomplete$zipcode)
hist(survey_incomplete$credit)

qqnorm(survey_incomplete$credit) #normal quantile plot. 
qqplot(survey_incomplete$salary, survey_incomplete$age, plot.it = TRUE) # q-q plot. 
# check for missing values 
anyNA(survey_incomplete)
is.na(survey_incomplete)


##########################################
# Preprocess Data and Feature Engineering
##########################################

# Change variable types 
str(complete_responses)
complete_responses$elevel  <- as.ordered(complete_responses$elevel)
complete_responses$car     <- as.factor(complete_responses$car)
complete_responses$zipcode <- as.factor(complete_responses$zipcode)
complete_responses$brand   <- as.factor(complete_responses$brand)
str(complete_responses)

str(survey_incomplete)
survey_incomplete$elevel  <- as.ordered(survey_incomplete$elevel)
survey_incomplete$car     <- as.factor(survey_incomplete$car)
survey_incomplete$zipcode <- as.factor(survey_incomplete$zipcode)
str(survey_incomplete)

# remove label column from prediction/new data - we'll predict this
survey_incomplete$brand <- NULL

# Split into training/val set and test set. We will use k-fold cross validation
# when training, so training and validation examples will both be pulled from 
# train_set 
in_training <- createDataPartition(complete_responses$brand, p=0.8, list=FALSE)
train_set <- complete_responses[in_training,]   
test_set <- complete_responses[-in_training,]   

# feature scaling
scale_params_train <- preProcess(train_set[, c(1,2,6)], 
                               method = c("center", "scale"))
print(scale_params_train) 

train_set <- predict(scale_params_train, train_set)
# scaled with train set means and std. devs.
test_set <- predict(scale_params_train, test_set) 
# scaled with train set means and std. devs.
prediction_set <- predict(scale_params_train, survey_incomplete) 

# save data
write.csv(train_set, "train_set.csv", row.names = FALSE)
write.csv(test_set, "test_set.csv", row.names = FALSE)
write.csv(prediction_set, "prediction_set.csv", row.names = FALSE)

# load data
train_set      <- read.csv("train_set.csv")
test_set       <- read.csv("test_set.csv")
prediction_set <- read.csv("prediction_set.csv")
# If reading from csv, need to change variable types again since read.csv() 
# imports all values as numeric.
train_set$elevel  <- as.ordered(train_set$elevel)
train_set$car     <- as.factor(train_set$car)
train_set$zipcode <- as.factor(train_set$zipcode)
train_set$brand   <- as.factor(train_set$brand)

test_set$elevel  <- as.ordered(test_set$elevel)
test_set$car     <- as.factor(test_set$car)
test_set$zipcode <- as.factor(test_set$zipcode)
test_set$brand   <- as.factor(test_set$brand)

prediction_set$elevel  <- as.ordered(prediction_set$elevel)
prediction_set$car     <- as.factor(prediction_set$car)
prediction_set$zipcode <- as.factor(prediction_set$zipcode)
str(train_set)
str(test_set)
str(prediction_set)



#################
# Train model(s)
#################

## ------- Random Forest ------- ##

# 10-fold cross validation. 
# summaryFunction: classification -> Data not skewed -> Use accuracy and kappa 
ranger_fit_control <- trainControl(method = 'cv', number = 10,
                             summaryFunction = defaultSummary)

# hyperparameter values to try
modelLookup('ranger')
ranger_grid <- expand.grid(mtry = c(2,4,6,8,10),
                          splitrule = c('gini'),
                          min.node.size = c(7,9))
nrow(ranger_grid)

# Try different sets of hyperparameters, pick the best set based on best 
# cross-validated accuracy (and kappa), and fit the final model to all the 
# training data using the optimal hyperparameter set.
tic <- Sys.time()
ranger_fit <- train(x = train_set[ ,1:6], y = train_set$brand,
                   method = "ranger",
                   trControl = ranger_fit_control,
                   tuneGrid = ranger_grid)
toc <- Sys.time()
runtime <- toc - tic

print(ranger_fit) 

# confusion matrix for the hold-out samples
confusionMatrix(ranger_fit, norm = "none") 
# cross-validated accuracy and kappa - accuracy 0.921, kappa 0.833

ggplot(ranger_fit)
ggplot(ranger_fit, metric = "Kappa")
varImp(ranger_fit) # most important features

print(ranger_fit$finalModel)  
# OOB accuracy  - OOB accuracy 1-0.079 = 0.921 

train_Pred <- predict(ranger_fit, train_set[ ,1:6])
confusionMatrix(data = train_Pred, reference = train_set$brand) 
# trainingset accuracy and kappa - accuracy 0.970, kappa 0.933

test_Pred <- predict(ranger_fit, test_set[ ,1:6])
confusionMatrix(data = test_Pred, reference = test_set$brand) 
# testset accuracy and kappa - accuracy 0.973, kappa 0.942

# save final model
#saveRDS(ranger_fit, "ranger_fit.rds") 

# load final model
ranger_fit <- readRDS("ranger_fit.rds")


##################
# Predict new data
##################

# predict with best model
prediction_pred <- predict(ranger_fit, prediction_set)
survey_incomplete$brand <- prediction_pred
write.csv(survey_incomplete, "predicted.csv")
summary(prediction_pred)
# In the incomplete survey, we predict that 1915 of those surveyed would have 
# prefered Acer while 3085 would have prefered Sony.

# stop cluster when done
stopCluster(cl)