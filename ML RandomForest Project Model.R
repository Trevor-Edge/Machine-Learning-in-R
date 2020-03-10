## Random Forest ML Project Model ##
rm(list = ls())


#Parallel Processing
#install.packages("doParallel")
library(doParallel)

cores=4
cl = makeCluster(cores)
registerDoParallel(cores)
getDoParWorkers()

# Training and Validation #
setwd("C:\\Users\\trevo.DESKTOP-Q3G2N9L\\Documents\\Data Mining & Machine Learning\\DM&ML Data\\")
train <- read.csv("TrainingSet.csv")
valid <- read.csv("TestingSet.csv")
test <- read.csv("Test_standardized.csv")

#install.packages("randomForest")
#install.packages("pROC")
library(pROC)
library(randomForest)

set.seed(123456)

# Subset of training dataset for faster training
train$target1 <- as.character(train$target1)
train$target1 <- as.factor(train$target1)
train$target2 <- as.character(train$target2)
train$target2 <- as.factor(train$target2)

#Optimal parameters mtry = 19 & ntree = 71
#Creating the Model
rf = randomForest(Target1 ~ N1+P2+K3+
                  Y2+C1+J3+C4+N6+D1+D5+R5+B6+N2+J6+H4+M6+E3+W6+M1+C5+D4+V6+E1+U6+P3+M3+S1+R2+R3+T5+H6+U5+V5+H1+J2+
                  B1+B5+X5+D6+I2+K4+A1+F3+Z5+G5+Z4+H2+L4+R4+A5+H5+D2+L5+J5+N3+M4+E6+I3+K2+Y4, 
                  data=train, ntree=71, mtry = 19, type='class')#, na.action=na.omit)
rf2 = randomForest(Target2 ~ N1+P2+K3+
                    Y2+C1+J3+C4+N6+D1+D5+R5+B6+N2+J6+H4+M6+E3+W6+M1+C5+D4+V6+E1+U6+P3+M3+S1+R2+R3+T5+H6+U5+V5+H1+J2+
                    B1+B5+X5+D6+I2+K4+A1+F3+Z5+G5+Z4+H2+L4+R4+A5+H5+D2+L5+J5+N3+M4+E6+I3+K2+Y4, 
                  data=train, ntree=71, mtry = 19, type='class')
varImpPlot(rf2)

# Validation set assessment ROC curves and AUC
# Needs to import ROCR package for ROC curve plotting:
library(ROCR)

# Calculate the probability of new observations belonging to each class
# prediction_for_roc_curve will be a matrix with dimensions data_set_size 2 number_of_classes
prediction_for_roc_curve <- predict(rf,valid[,-153],type="prob")
prediction_for_roc_curve2 <- predict(rf2,valid[,-153],type="prob")
prediction_for_roc_curve3 <- predict(rf,test,type="prob")
prediction_for_roc_curve4 <- predict(rf2, test, type="prob")

# Use pretty colors:
pretty_colors <- c("#F8766D")

# Specify the different classes 
classes <- levels(valid$Target1)

# For each class
# Define which observations belong to class[i]
true_values <- ifelse(valid[,153]==classes[2],1,0)

# Assess the performance of classifier for class[i]
pred <- prediction(prediction_for_roc_curve[,2],true_values)
pred2 <- prediction(prediction_for_roc_curve2[,2],true_values)

perf <- performance(pred, "tpr", "fpr")
perf2 <- performance(pred2, "tpr", "fpr")

plot(perf,main="ROC Curve",col=pretty_colours[1])
plot(perf2,main="ROC Curve",col=pretty_colours[1])

# Calculate the AUC and print it to screen
auc.perf <- performance(pred, measure = "auc")
auc.perf2 <- performance(pred2, measure = "auc")

print(auc.perf@y.values)
print(auc.perf2@y.values)

## AUC for Target1 = 0.7744518 ##
## AUC for Target2 = 0.7314334 ##
rf$confusion

rf$err.rate
head(rf$err.rate)
rf$err.rate[71,]

# Test on Validation
#We should check our random forest model on the validation model as a final
# test of performance and screen for overfitting
#Validation Misclassification Rate: 0.23385
vscores = predict(rf, valid, type='class')
cat('Validation Misclassification Rate:', sum(vscores!=valid$Target1)/nrow(valid))

#Validation Misclassification Rate: 0.25805
vscores2 = predict(rf2, valid, type='class')
cat('Validation Misclassification Rate:', sum(vscores2!=valid$Target1)/nrow(valid))

#' Let's try to get a sense of how many trees are necessary to get a good 
#' performance on the test data. (Tuning the hyperparameter ntrees.)
# About 40-80 trees is optimal (80 being the most optimal at the least trees)
accuracy=vector()
ntrees=seq(1,400,20)
i=1
for(n in ntrees){
  rf = randomForest(Target1 ~ N1+P2+K3+Y2+C1+J3+C4+N6+D1+D5+R5+B6+N2+J6+H4+M6+E3+W6+M1+C5+D4+V6+
                      E1+U6+P3+M3+S1+R2+R3+T5+H6+U5+V5+H1+J2+B1+B5+X5+D6+I2+K4+A1+F3+Z5+G5+Z4+H2+
                      L4+R4+A5+H5+D2+L5+J5+N3+M4+E6+I3+K2+Y4, data=train, ntree=n, type='class', na.action = na.omit)
  test_pred =  predict(rf,valid,type='class')
  accuracy[i] =  sum(test_pred!=valid$Target1)/nrow(valid)
  i=i+1
}
plot(ntrees, accuracy)

#' Clear that we don't need that many trees here. Effect seems to level off
#' after 20-100 trees. Let's look over that range.
#' # 71 trees is the optimal amount
accuracy=vector()
ntrees=seq(1,100,5)
i=1
for(n in ntrees){
  rf = randomForest(Target1 ~ N1+P2+K3+
                      Y2+C1+J3+C4+N6+D1+D5+R5+B6+N2+J6+H4+M6+E3+W6+M1+C5+D4+V6+E1+U6+P3+M3+S1+R2+R3+T5+H6+U5+V5+H1+J2+
                      B1+B5+X5+D6+I2+K4+A1+F3+Z5+G5+Z4+H2+L4+R4+A5+H5+D2+L5+J5+N3+M4+E6+I3+K2+Y4, 
                    data=train, ntree=n, type='class')
  test_pred =  predict(rf,valid,type='class')
  accuracy[i] =  sum(test_pred!=valid$Target1)/nrow(valid)
  i=i+1
}
plot(ntrees, accuracy)

#' We could also play around with the other parameters, like mtry:
# Around 19 mtry is optimal
accuracy=vector()
mtry=seq(16,21)
i=1
for(m in mtry){
  rf = randomForest(Target1 ~ N1+P2+K3+Y2+C1+J3+C4+N6+D1+D5+R5+B6+N2+J6+H4+M6+E3+
                      W6+M1+C5+D4+V6+E1+U6+P3+M3+S1+R2+R3+T5+H6+U5+V5+H1+J2+B1+B5+
                      X5+D6+I2+K4+A1+F3+Z5+G5+Z4+H2+L4+R4+A5+H5+D2+L5+J5+N3+M4+E6+I3+K2+Y4, 
                    data=train,mtry=m, ntree=71, type='class', na.action = na.omit)
  test_pred =  predict(rf,valid,type='class')
  accuracy[i] =  sum(test_pred!=valid$Target1)/nrow(valid)
  i=i+1
}
plot(mtry, accuracy)

#' Tells us which variables are important in prediction of each digit
importance(rf)

#Ending Parallel Processing
stopCluster(cl)

