#################
# Project Summary
#################

# Datasets:     existingproductattributes2017.csv (training/validation/test set), 
#               newproductattributes2017.csv (prediction set), and 
#               predictions.csv (predictions made on newproductattributes2017)
#
# Final Model:  lm_fit.rds (multiple linear regression).
#
# Scenario:     Electronidex is a mock online and brick-and-mortar electronics 
#               retailer. We have 24 new items we would like to add to the store.
#               Using data on existing products, such as the price, the number of
#               5-star reviews, profit margin, shipping dimensions, sales volume,
#               and so forth, we would like to predict the sales volume of the 
#               products in the new lineup.
#
# Goal:         Predict the sales volume in newproductattributes2017.csv   
#
# Conclusion:  Top performing model: multiple linear regression. Only the 
#              intercept term and the coefficient for x5StarReviews feature were
#              statistically significant (P-value < 0.001). Further, the sales
#              volume was almost a perfectly linear function of the number of
#              5 star reviews, as shown by the very small RMSE values.
#
#              cross-validation RMSE: 1.52e-12
#              train set RMSE: 1.54e-12
#              test set RMSE: 5.06e-13
#
#              The model adequacy was verified by looking at the residuals both
#              on the training and the test set.
#              
# Reason:      Very low cross-validation RMSE and RMSE on the test set. Variance
#              is not that high due to gap between training set RMSE and cross-
#              validation RMSE being small.

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
existing_products <- read.csv("existingproductattributes2017.csv", 
                              stringsAsFactors = FALSE, header=T)

# prediction/new data
new_products <- read.csv("newproductattributes2017.csv", 
                        stringsAsFactors = FALSE, header=T)


################
# Evaluate data
################

#--- training/validation/test set ---#
str(existing_products)  # 80 obs. of 18 variables 
summary(existing_products)

# plots
barplot(table(existing_products$ProductType))
hist(existing_products$Price)
hist(existing_products$x5StarReviews)
hist(existing_products$x4StarReviews)
hist(existing_products$x3StarReviews)
hist(existing_products$x2StarReviews)
hist(existing_products$x1StarReviews)
hist(existing_products$PositiveServiceReview)
hist(existing_products$NegativeServiceReview)
hist(existing_products$Recommendproduct)
hist(existing_products$BestSellersRank)
hist(existing_products$ShippingWeight)
hist(existing_products$ProductDepth)
hist(existing_products$ProductWidth)
hist(existing_products$ProductHeight)
hist(existing_products$ProfitMargin)
hist(existing_products$Volume)

plot(existing_products$Price, existing_products$Volume)
plot(existing_products$x5StarReviews, existing_products$Volume) # linear
plot(existing_products$x4StarReviews, existing_products$Volume)
plot(existing_products$x3StarReviews, existing_products$Volume)
plot(existing_products$x2StarReviews, existing_products$Volume)
plot(existing_products$x1StarReviews, existing_products$Volume)
plot(existing_products$PositiveServiceReview, existing_products$Volume)
plot(existing_products$NegativeServiceReview, existing_products$Volume)
plot(existing_products$Recommendproduct, existing_products$Volume)
plot(existing_products$BestSellersRank, existing_products$Volume)
plot(existing_products$ShippingWeight, existing_products$Volume)
plot(existing_products$ProductDepth, existing_products$Volume)
plot(existing_products$ProductWidth, existing_products$Volume)
plot(existing_products$ProductHeight, existing_products$Volume)
plot(existing_products$ProfitMargin, existing_products$Volume)

# check for collinearity - correlation matrix
str(existing_products)
corrAll = cor(existing_products[ ,c(3:11, 13:18)], use = "all.obs", method = "pearson")
corrplot(corrAll, order = "hclust") # sorts based on level of collinearity
corrplot(corrAll, method = "circle")
print(corrAll)

# check for missing values
anyNA(existing_products)
is.na(existing_products)
# BestSellersRank has missing values


#--- prediction/new data ---#
str(new_products)  # 24 obs. of 18 variables 
summary(new_products)

# plot
barplot(table(new_products$ProductType))
hist(new_products$Price)
hist(new_products$x5StarReviews)
hist(new_products$x4StarReviews)
hist(new_products$x3StarReviews)
hist(new_products$x2StarReviews)
hist(new_products$x1StarReviews)
hist(new_products$PositiveServiceReview)
hist(new_products$NegativeServiceReview)
hist(new_products$Recommendproduct)
hist(new_products$BestSellersRank)
hist(new_products$ShippingWeight)
hist(new_products$ProductDepth)
hist(new_products$ProductWidth)
hist(new_products$ProductHeight)
hist(new_products$ProfitMargin)
hist(new_products$Volume)


# check for missing values 
anyNA(new_products)
is.na(new_products)
# no missing values

##########################################
# Preprocess Data and Feature Engineering
##########################################

#--- training/validation/test set ---#

# BestSellersRank contains missing data. Gere we will simply remove the attribute.
existing_products$BestSellersRank <- NULL

# Remove ID attribute(s)
existing_products$ProductNum <- NULL

# Address multicollinearity 
corrAll = cor(existing_products, use = "all.obs", method = "pearson")
write.csv(corrAll, "existing_products_corr_matrix.csv")
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

existing_products$x4StarReviews <- NULL
existing_products$x2StarReviews <- NULL
existing_products$x1StarReviews <- NULL
str(existing_products)

corr13v = cor(existing_products[, c(2:13)], use = "all.obs", method = "pearson")
write.csv(corr13v, "corr_matrix_13v.csv")
corrplot(corr13v, order = "hclust") # sorts based on level of collinearity
corrplot(corr13v, method = "circle")

# Change variable types 
str(existing_products)
existing_products$ProductType <- as.factor(existing_products$ProductType)
unique(existing_products$ProductType)

# Encode all categorical variables to numeric data as dummy variables. 
# No linear dependencies: fullRank = TRUE.
dummies <- dummyVars(~., existing_products, fullRank = TRUE)
existing_products <- data.frame(predict(dummies, existing_products))
str(existing_products)

# Split into training/val set and test set. We will use k-fold cross validation
# when training, so training and validation examples will both be pulled from 
# train_set 
in_training <- createDataPartition(existing_products$Volume, p=0.8, list=FALSE)
train_set <- existing_products[in_training,]   
test_set <- existing_products[-in_training,] 

# feature scaling
scale_params_train <- preProcess(train_set[, 12:22], 
                                 method = c("center", "scale"))
print(scale_params_train) 
train_set <- predict(scale_params_train, train_set)
test_set <- predict(scale_params_train, test_set)


# --- prediction/new set --#
new_products$ProductNum <- NULL
new_products$x4StarReviews <- NULL
new_products$x2StarReviews <- NULL
new_products$x1StarReviews <- NULL
new_products$ProductType <- as.factor(new_products$ProductType)
dummies2 <- dummyVars(~., new_products, fullRank = TRUE)
new_products <- data.frame(predict(dummies2, new_products))
new_products <- predict(scale_params_train, new_products)
new_products$BestSellersRank <- NULL
prediction_set <- new_products


#--- save data ---#
write.csv(train_set, "train_set.csv", row.names = FALSE)
write.csv(test_set, "test_set.csv", row.names = FALSE)
write.csv(prediction_set, "prediction_set.csv", row.names = FALSE)

#--- load data ---#
train_set      <- read.csv("train_set.csv")
test_set       <- read.csv("test_set.csv")
prediction_set <- read.csv("prediction_set.csv")


#################
# Train model(s)
#################

## ------- Multiple Linear Regression ------- ##

# 10-fold cross validation. 
lm_fit_control <- trainControl(method = 'cv', number = 10,
                                   summaryFunction = defaultSummary)

tic <- Sys.time()
lm_fit <- train(x = train_set[ ,1:22], y = train_set$Volume,
                    method = 'lm',
                    trControl = lm_fit_control)
toc <- Sys.time()
runtime <- toc - tic

print(lm_fit) 
# cross-validation RMSE: 1.52e-12
varImp(lm_fit)
print(lm_fit$finalModel)  

train_pred <- predict(lm_fit, train_set[ ,1:22])
(sum(train_pred - train_set$Volume)^2/65)^0.5
# train set RMSE: 1.54e-12

test_pred <- predict(lm_fit, test_set[ ,1:22])
(sum(test_pred - test_set$Volume)^2/23)^0.5
# test set RMSE: 5.06e-13


# save final model
saveRDS(lm_fit, "lm_fit.rds") 

# load final model
lm_fit <- readRDS("lm_fit.rds")


######################
# Check Model Adequacy
######################
train_set_residuals = train_set$Volume - train_pred 
plot(train_pred, train_set_residuals) # relatively homoscedastic
hist(train_set_residuals)
qqnorm(train_set_residuals) # residuals are relatively normally distributed

test_set_residuals = test_set$Volume - test_pred
plot(test_pred, test_set_residuals) # relatively homoscedastic
hist(test_set_residuals)
qqnorm(test_set_residuals) # residuals are relatively normally distributed

# model adequacy - good!

##################
# Predict new data
##################

# make predictions
prediction_pred <- predict(lm_fit, prediction_set)
newproductattributes2017 <- read.csv("newproductattributes2017.csv", 
                                     stringsAsFactors = FALSE, header=T)
newproductattributes2017$Volume <- prediction_pred
write.csv(newproductattributes2017, "predicted_new_product_volumes.csv")


# stop cluster when done
stopCluster(cl)