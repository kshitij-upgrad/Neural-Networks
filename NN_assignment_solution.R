install.packages("h2o")
library(h2o)
h2o.init()

telecom_train <- h2o.importFile("telecom_nn_train.csv")
telecom_test <- h2o.importFile("telecom_nn_test.csv")
set.seed(100)
telecom_train$Churn <- as.factor(telecom_train$Churn)
s = h2o.splitFrame(telecom_train, ratios = 0.7)

telecom_train1 <- s[[1]]
validation_data <- s[[1]]


#telecom_train[,-10] <- scale(telecom_train[,-10])
# Withut Epochs
nnet <- h2o.deeplearning(names(telecom_train[, -10]), names(telecom_train1[, 10]),
                         telecom_train, distribution = "bernoulli", validation_frame = validation_data,
                         activation = "Rectifier", hidden = c(800,800,800,800,800), epochs = 1,
                         standardize = TRUE, rate = 0.0001, loss = "CrossEntropy", nfolds = 5 )

prediction <- h2o.predict(nnet, telecom_test[,-10])

prediction2 <- cbind(as.data.frame(prediction[,1]) , as.data.frame(telecom_test[,10]))
confusionMatrix(prediction2[, 1], prediction2[, 2], positive = 'Yes')                    


#With Epochs

nnet <- h2o.deeplearning(names(telecom_train[, -10]), names(telecom_train1[, 10]),
                         telecom_train, distribution = "bernoulli", validation_frame = validation_data,
                         activation = "Rectifier", hidden = c(100,100,100,100,100), epochs = 6,
                         standardize = TRUE, rate = 0.0001, loss = "CrossEntropy", nfolds = 5, hidden_dropout_ratios = c(0.1, 0.1, 0.1, 0.1, 0.1) )

prediction <- h2o.predict(nnet, telecom_test[,-10])
prediction2 <- cbind(as.data.frame(prediction[,1]) , as.data.frame(telecom_test[,10]))
confusionMatrix(prediction2[, 1], prediction2[, 2], positive = 'Yes')  