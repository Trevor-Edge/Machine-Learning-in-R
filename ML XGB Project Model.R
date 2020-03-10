############################
## Gradient Boosting in R ##
##      TREVOR EDGE       ##
##    Machine Learning    ##
############################
rm(list=ls())

#Parallel Processing
#install.packages("doParallel")
library(doParallel)

#Parallel multi-core processing #
cores=4
cl = makeCluster(cores)
registerDoParallel(cores)
getDoParWorkers()

#Preparing the Data
# Training and Validation #
setwd("C:\\Users\\trevo.DESKTOP-Q3G2N9L\\Documents\\Data Mining & Machine Learning\\DM&ML Data\\")
train <- read.csv("TrainingSet.csv")
valid <- read.csv("TestingSet.csv")

train$Target3[train$Target1 == "True"] <- "1"
train$Target3[train$Target1 == "False"] <- "0"
valid$Target3[valid$Target1 == "True"] <- "1"
valid$Target3[valid$Target1 == "False"] <- "0"

train$Target4[train$Target2 == "True"] <- "1"
train$Target4[train$Target2 == "False"] <- "0"
valid$Target4[valid$Target2 == "True"] <- "1"
valid$Target4[valid$Target2 == "False"] <- "0"

train$Target3 <- as.numeric(train$Target3)
train$Target4 <- as.numeric(train$Target4)
valid$Target3 <- as.numeric(valid$Target3)
valid$Target4 <- as.numeric(valid$Target4)

# select variables
myvars <- c("N1","P2","K3","Y2","C1","J3","C4","N6","D1","D5","R5","B6","N2","J6","H4","M6","E3","W6","M1","C5","D4","V6","E1",
            "U6","P3","M3","S1","R2","R3","T5","H6","U5","V5","H1","J2","B1","B5","X5","D6","I2","K4","A1","F3","Z5","G5","Z4","H2","L4",
            "R4","A5","H5","D2","L5","J5","N3","M4","E6","I3","K2","Y4","Target3")
train2 <- train[myvars]
valid2 <- valid[myvars]
set.seed(1234567)
train2 <- train[sample(nrow(train), 30000), ]

train$Target3 <- as.factor(train$Target3)
valid$Target3 <- as.factor(valid$Target3)
train$Target4 <- as.factor(train$Target4)
valid$Target4 <- as.factor(valid$Target4)
#install.packages("Matrix")
library(Matrix)

#Variables chosen by importance from random forest (top 60 variables)
#xtrain = model.matrix(Target3 ~ ., data=train2)
#xvalid = model.matrix(Target3 ~ ., data=valid)

#XTRAIN
xtrain = model.matrix(Target3 ~ N1+P2+K3+
                        Y2+C1+J3+C4+N6+D1+D5+R5+B6+N2+J6+H4+M6+E3+W6+M1+C5+D4+V6+E1+U6+P3+M3+S1+R2+R3+T5+H6+U5+V5+H1+J2+
                        B1+B5+X5+D6+I2+K4+A1+F3+Z5+G5+Z4+H2+L4+R4+A5+H5+D2+L5+J5+N3+M4+E6+I3+K2+Y4, data=train)
xtrain2 = model.matrix(Target4 ~ N1+P2+K3+
                        Y2+C1+J3+C4+N6+D1+D5+R5+B6+N2+J6+H4+M6+E3+W6+M1+C5+D4+V6+E1+U6+P3+M3+S1+R2+R3+T5+H6+U5+V5+H1+J2+
                        B1+B5+X5+D6+I2+K4+A1+F3+Z5+G5+Z4+H2+L4+R4+A5+H5+D2+L5+J5+N3+M4+E6+I3+K2+Y4, data=train)
#XVALID
xvalid = model.matrix(Target3 ~ N1+P2+K3+
                       Y2+C1+J3+C4+N6+D1+D5+R5+B6+N2+J6+H4+M6+E3+W6+M1+C5+D4+V6+E1+U6+P3+M3+S1+R2+R3+T5+H6+U5+V5+H1+J2+
                       B1+B5+X5+D6+I2+K4+A1+F3+Z5+G5+Z4+H2+L4+R4+A5+H5+D2+L5+J5+N3+M4+E6+I3+K2+Y4, data=valid)
xvalid2 = model.matrix(Target4 ~ N1+P2+K3+
                        Y2+C1+J3+C4+N6+D1+D5+R5+B6+N2+J6+H4+M6+E3+W6+M1+C5+D4+V6+E1+U6+P3+M3+S1+R2+R3+T5+H6+U5+V5+H1+J2+
                        B1+B5+X5+D6+I2+K4+A1+F3+Z5+G5+Z4+H2+L4+R4+A5+H5+D2+L5+J5+N3+M4+E6+I3+K2+Y4, data=valid)
#YTRAIN
ytrain = as.numeric(levels(train$Target3))[train$Target3]
ytrain2 = as.numeric(levels(train$Target4))[train$Target4]

#YVALID
yvalid = as.numeric(levels(valid$Target3))[valid$Target3]
yvalid2 = as.numeric(levels(valid$Target4))[valid$Target4]

as.factor(levels(train2$Target3))[train2$Target3]
#Creating the Model
#install.packages("xgboost")
library(xgboost)
set.seed(1234567)

#run the model assuming parameters are already tuned
xgb <- xgboost(data = xtrain,
               label = ytrain,
               eta = 0.9,
               max_depth = 11,
               gamma = 0,
               nround = 100,
               subsample = 0.75,
               colsample_bylevel = 0.75,
               num_class = 2,
               objective = "multi:softmax",
               nthread = 3,
               eval_metric = "merror",
               verbose = 0)

xgb2 <- xgboost(data = xtrain2,
               label = ytrain2,
               eta = 0.9,
               max_depth = 11,
               gamma = 0,
               nround = 100,
               subsample = 0.75,
               colsample_bylevel = 0.75,
               num_class = 2,
               objective = "multi:softmax",
               nthread = 3,
               eval_metric = "merror",
               verbose = 0)

#Use the predict() function to check the model's performance on 
#validation data
ptrain = predict(xgb, xtrain)
pvalid = predict(xgb, xvalid)
cat('Confusion Matrix:')
table(pvalid, valid$Target3)

ptrain2 = predict(xgb2, xtrain2)
pvalid2 = predict(xgb2, xvalid2)
cat('Confusion Matrix:')
table(pvalid2, valid$Target4)

XGBmrt = sum(ptrain != train$Target3)/length(train$Target3)
XGBmrv = sum(pvalid != valid$Target3)/length(valid$Target3)
cat('XGB Training Misclassification Rate:', XGBmrt)
cat('XGB Validation Misclassification Rate:', XGBmrv)

XGBmrt2 = sum(ptrain2 != train$Target4)/length(train$Target4)
XGBmrv2 = sum(pvalid2 != valid$Target4)/length(valid$Target4)
cat('XGB Training Misclassification Rate:', XGBmrt2)
cat('XGB Validation Misclassification Rate:', XGBmrv2)

importance <- xgb.importance(feature_names = colnames(xtrain), 
                             model = xgb)
importance2 <- xgb.importance(feature_names = colnames(xtrain2), 
                             model = xgb2)
head(importance)
head(importance2)

max(xgb$evaluation_log$train_auc)
max(xgb2$evaluation_log$train_auc)
## 3 TUNING THE MODEL ##
## ALTER 1 PARAMETER AT A TIME, KEEPING ALL OTHERS CONSTANT TO ISOLATE TUNING EFFECTS ##
## Following parameter values for the default model (ie. when the parameters are held constant) ##
# eta = 0.1 #
# colsample_bylevel = 0.67 #
# max_depth = 5 #
# sub_sample = 0.75 #

#3.1 Tuning Eta
#install.packages("ggplot2")
#install.packages("reshape2")
library(reshape2)
library(ggplot2)
########################################
####### PARAMETERS TO BE SEARCHED ######
########################################

# Eta candidates
eta = c(0.5, 0.6, 0.7, 0.8, 0.9, .95)

# colsample_bylevel candidates
cs = c(.33, .67, 1)

# max_depth candidates
md = c(8,9, 10, 11)

#sub_sample candidates
ss = c( .5, .65, .75, .85)

# fixed number of rounds
num_rounds = 100

###########################
#coordinates of default model in terms of
#the entries in the the vectors above:
default = c(2,2,2,3)

#3.2 Tuning Eta - #########USE 0.95 for ETA############
#create empty matrices to hold the convergence
#and prediction results for our search over eat:
conv_eta = matrix(NA, num_rounds, length(eta))
pred_eta = matrix(NA, dim(valid)[1], length(eta))
word = rep('eta', length(eta))
colnames(conv_eta) = colnames(pred_eta) = paste(word,eta)
for(i in 1:length(eta)){
  params=list(eta = eta[i], colsample_bylevel = cs[default[2]],
              subsample = ss[default[4]], max_depth = md[default[3]],
              min_child_weight = 1)
  xgb=xgboost(xtrain, label = ytrain, nrounds = num_rounds, params = params, 
              verbose = 0, num_class = 10, objective = "multi:softmax")
  conv_eta[,i] = xgb$evaluation_log$train_merror
  pred_eta[,i] = predict(xgb, xvalid)
}

cat('Validation Misclassification Error for Each eta:')

(1-colMeans(yvalid == pred_eta))

## Reshape the data frame so that the eta value is a variable
## rather than having a column for each eta value:
conv_eta = data.frame(iter=1:num_rounds, conv_eta)
conv_eta2 = melt(conv_eta, id.vars = "iter", value.name = 'MisclassificationRate', variable.name = 'eta')
ggplot(data = conv_eta2) + 
  geom_line(aes(x = iter, y = MisclassificationRate, color = eta)) +
  labs(title = "Convergence on Training for each Eta")

## Misclassification rate seems to be improving (along with the convergence
## rate) as we increase eta towards the upper range of our sample. Eta at 0.9
## produces the best results on validation.

#3.3 Tuning colsample_bylevel
# create empty matrices to hold the convergence
# and prediction results for our search over
# colsample_bylevel:
conv_cs = matrix(NA,num_rounds, length(cs))
pred_cs = matrix(NA,dim(valid)[1], length(cs))
word = rep('cs', length(cs))
colnames(conv_cs) = colnames(pred_cs) = paste(word,cs)
for(i in 1:length(cs)){
  params=list(cs = cs[i], eta=eta[default[i]],
              subsample = ss[default[4]], max_depth = md[default[3]],
              min_child_weight = 1)
  xgb=xgboost(xtrain, label = ytrain, nrounds = num_rounds, params = params, verbose = 0, 
              num_class = 10, objective = "multi:softmax")
  conv_cs[,i] = xgb$evaluation_log$train_merror
  pred_cs[,i] = predict(xgb, xvalid)
}
cat('Validation Misclassification Error for Each colsample_bylevel:')
(Misclass_cs = 1-colMeans(yvalid == pred_cs))

## Reshape the data frame so that the eta value is a variable - NO REAL DIFFERENCE IN COLSAMPLE PARAMETER#
## rather than having a column for each eta value:
conv_cs = data.frame(iter=1:num_rounds, conv_cs)
conv_cs2 = melt(conv_cs, id.vars = "iter", value.name = 'MisclassificationRate', variable.names = 'cs')
colnames(conv_cs2)[colnames(conv_cs2)=="variable"] <- "cs"
ggplot(data = conv_cs2) +
  geom_line(aes(x = iter, y = MisclassificationRate, color = cs)) +
  labs(title = "Convergence on Training for each colsample_bylevel")

#3.4 Tuning subsample - USE .75 FOR SUBSAMPLE #
# create empty matrices to hold the convergence
# and prediction results for our search over
# subsample:
conv_ss = matrix(NA,num_rounds, length(ss))
pred_ss = matrix(NA,dim(valid)[1], length(ss))
word = rep('ss', length(ss))
colnames(conv_ss) = colnames(pred_ss) = paste(word,ss)
for(i in 1:length(ss)){
  params=list(cs = cs[default[2]], eta=eta[default[1]],
              subsample = ss[i], max_depth = md[default[3]],
              min_child_weight = 1)
  xgb=xgboost(xtrain, label = ytrain, nrounds = num_rounds, params = params, verbose = 0, 
              num_class = 10, objective = "multi:softmax")
  conv_ss[,i] = xgb$evaluation_log$train_merror
  pred_ss[,i] = predict(xgb, xvalid)
}
cat('Validation Misclassification Error for Each subsample:')
(Misclass_ss = 1-colMeans(ytest == pred_md))

## Reshape the data frame so that the eta value is a variable
## rather than having a column for each eta value:
conv_ss = data.frame(iter=1:num_rounds, conv_ss)
conv_ss2 = melt(conv_ss, id.vars = "iter", value.name = 'MisclassificationRate', variable.name = 'ss')
colnames(conv_cs2)[colnames(conv_cs2)=="variable"] <- "cs"
ggplot(data = conv_ss2) +
  geom_line(aes(x = iter, y = MisclassificationRate, color = ss)) +
  labs(title = "Convergence on Training for each subsample")

#3.5 Tuning max_depth - USE 11 for MAX DEPTH
# create empty matrices to hold the convergence
# and prediction results for our search over
# max_depth:
conv_md = matrix(NA,num_rounds, length(md))
pred_md = matrix(NA,dim(valid)[1], length(md))
word = rep('md', length(md))
colnames(conv_md) = colnames(pred_md) = paste(word,md)
for(i in 1:length(md)){
  params=list(cs = cs[default[2]], eta=eta[default[1]],
              subsample = ss[default[3]], max_depth = md[default[i]],
              min_child_weight = 1)
  xgb=xgboost(xtrain, label = ytrain, nrounds = num_rounds, params = params, verbose = 0, 
              num_class = 10, objective = "multi:softmax")
  conv_md[,i] = xgb$evaluation_log$train_merror
  pred_md[,i] = predict(xgb, xvalid)
}
cat('Validation Misclassification Error for Each max_depth:')
(Misclass_cs = 1-colMeans(ytest == pred_md))

## Reshape the data frame so that the eta value is a variable
## rather than having a column for each eta value:
conv_md = data.frame(iter=1:num_rounds, conv_md)
conv_md2 = melt(conv_md, id.vars = "iter", value.name = 'MisclassificationRate', variable.name = 'md')
colnames(conv_cs2)[colnames(conv_cs2)=="variable"] <- "cs"
ggplot(data = conv_md2) +
  geom_line(aes(x = iter, y = MisclassificationRate, color = md)) +
  labs(title = "Convergence on Training for each max_depth")

#Ending multi-core processing
stopCluster(cl)


