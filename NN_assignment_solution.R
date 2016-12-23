library(h2o)
library(caTools)
library(caret)
h2o.init()

telecom <- read.csv("telecom_nn_train.csv")
str(telecom)

# Dividing 70-30 into training and validation respectively; using sample.split to ensure the same class ratio
split_indices <- sample.split(telecom$Churn, SplitRatio = 0.70)
telecom_train <- telecom[split_indices, ]
telecom_validation <- telecom[!split_indices, ]
write.csv(telecom_train, "telecom_train.csv", row.names = F)
write.csv(telecom_validation, "telecom_validation.csv", row.names = F)



telecom_train <- h2o.importFile("telecom_train.csv")
telecom_validation <- h2o.importFile("telecom_validation.csv")

# Converting Churn into factor
telecom_train$Churn <- as.factor(telecom_train$Churn)

set.seed(100)

# 1. Training Without Epochs 
## Experiment 1.1: With 3 hidden layers with 300 neurons each and dropout ratio of 10% for each layer
nnet_1 <- h2o.deeplearning(names(telecom_train[, -10]), 
                         names(telecom_train[, 10]),
                         training_frame = telecom_train, 
                         validation_frame = telecom_validation,
                         distribution = "bernoulli", 
                         activation = "RectifierWithDropout", 
                         hidden = c(300,300,300), 
                         hidden_dropout_ratios = c(0.1, 0.1, 0.1), 
                         epochs = 1,
                         standardize = TRUE) 
           
nnet_1

# In nnet_1, the error rates on training and validation sets are about 23% and 25% respectively    

## Experiment 1.2: With an extra hidden layer
nnet_2 <- h2o.deeplearning(names(telecom_train[, -10]), 
                         names(telecom_train[, 10]),
                         training_frame = telecom_train, 
                         validation_frame = telecom_validation,
                         distribution = "bernoulli", 
                         activation = "RectifierWithDropout", 
                         hidden = c(300,300,300, 300), 
                         hidden_dropout_ratios = c(0.1, 0.1, 0.1, 0.1), 
                         epochs = 1,
                         standardize = TRUE) 

nnet_2
# training and validation error are approx 22.2% and 20.9% respectively; a slight improvement 
# over the previous one
# let's try out a different activation function

## Experiment 1.3: With Tanh activation function
nnet_3 <- h2o.deeplearning(names(telecom_train[, -10]), 
                         names(telecom_train[, 10]),
                         training_frame = telecom_train, 
                         validation_frame = telecom_validation,
                         distribution = "bernoulli", 
                         activation = "TanhWithDropout", 
                         hidden = c(300,300,300, 300), 
                         hidden_dropout_ratios = c(0.1, 0.1, 0.1, 0.1), 
                         epochs = 1,
                         standardize = TRUE) 

nnet_3
# The training and validation error are approx 25.8% and 21.5% respectively; an increase in both errors
# compared to the previous model
# Let's stick with RectifierWithDropout and use epochs to train the network

#2. Training With Epochs
## Experiment 2.1: Using 6 epochs and other hyperparameters from experiment 1.2 

nnet_4 <- h2o.deeplearning(names(telecom_train[, -10]), 
                         names(telecom_train[, 10]),
                         training_frame = telecom_train,
                         validation_frame = telecom_validation,
                         distribution = "bernoulli", 
                         activation = "RectifierWithDropout", 
                         hidden = c(300,300,300,300), 
                         hidden_dropout_ratios = c(0.1, 0.1, 0.1, 0.1), 
                         epochs = 6,
                         standardize = TRUE)

nnet_4
# The training and validation error are approx 19.70% and 23.59% respectively

## Experiment 2.2: Using 10 epochs 
nnet_5 <- h2o.deeplearning(names(telecom_train[, -10]), 
                         names(telecom_train[, 10]),
                         training_frame = telecom_train,
                         validation_frame = telecom_validation,
                         distribution = "bernoulli", 
                         activation = "RectifierWithDropout", 
                         hidden = c(300,300,300,300), 
                         hidden_dropout_ratios = c(0.1, 0.1, 0.1, 0.1), 
                         epochs = 10,
                         standardize = TRUE)

nnet_5
# slight improvement in error rates compared to nnet_4

## Experiment 2.3: Using 20 epochs
nnet_6 <- h2o.deeplearning(names(telecom_train[, -10]), 
                         names(telecom_train[, 10]),
                         training_frame = telecom_train,
                         validation_frame = telecom_validation,
                         distribution = "bernoulli", 
                         activation = "RectifierWithDropout", 
                         hidden = c(300,300,300,300), 
                         hidden_dropout_ratios = c(0.1, 0.1, 0.1, 0.1), 
                         epochs = 20,
                         standardize = TRUE)



nnet_6
# not much improvement; sticking to nnet_5

nnet_final <- nnet_5

# Evaluate on validation data
prediction <- h2o.predict(nnet_5, telecom_validation[,-10])
prediction2 <- cbind(as.data.frame(prediction[,1]) , as.data.frame(telecom_validation[,10]))
confusionMatrix(prediction2[, 1], prediction2[, 2], positive = 'Yes')  
