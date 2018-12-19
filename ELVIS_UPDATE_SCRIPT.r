############################################################
#ELVIS DASHBOARD UPDATE SCRIPT
#CURRENT AS OF DECEMBER 2018
#CLAYTON BESAW - OEFR FORECASTING
############################################################
#JUST RUN SCRIPT AFTER UPDATING FILES IN ELVIS ELECTION SHEET DATABASE
#PACKAGES
library(caret)
library(data.table)
library(doParallel) 
library(reshape)
library(randomForest)
library(pROC)
library(caretEnsemble)
library(mlbench)
library(gbm)
library(party)
library(glmnet)
library(klaR)
library(stepPlr)
library(plyr)
library(pdp)
library(bartMachine)
library(caTools)
library(e1071)
library(ggplot2)
#SETUP
#import data and setup
neld2 <- read.csv("O:/Dropbox/Data/election_violence_forecast/ELVIS ELECTION SHEET DATABASE/elvis_master_dec2018.csv", header = TRUE)
# neld2$l.elecViolence1 <- as.factor(neld2$l.elecViolence1)
# neld2$l.elecViolence2 <- as.factor(neld2$l.elecViolence2)
#PCA?
pca1 <- prcomp( ~ l.elecViolence1 + tae1 + nvio1, 
                data = neld2, 
                scale = TRUE)
loadings <- as.data.frame(predict(pca1, newdata = neld2))
neld2$process1 <- loadings$PC1
neld2$process1_2 <- loadings$PC2
neld2$process1_3 <- loadings$PC3
pca2 <- prcomp(~ l.elecViolence2 + tae2 + nvio2,
               data = neld2,
               scale = TRUE)
loadings2 <- as.data.frame(predict(pca2, newdata = neld2))
neld2$process2 <- loadings2$PC1
neld2$process2_2 <- loadings2$PC2
neld2$process2_3 <- loadings2$PC3

#change target ordering
neld2$elecViolence1 <- factor(neld2$elecViolence1, levels = c("vio", "peace"))
neld2$elecViolence2 <- factor(neld2$elecViolence2, levels = c("vio", "peace"))

#make dates work for easy test/train split as time goes on
neld2$dates <- as.Date(neld2$dates, format = "%m/%d/%Y")
neld2 <- neld2[order(neld2$dates), ]

#train/test split for annual prediction
train_df <- neld2[neld2$year != 2018, ]
test_df <- neld2[neld2$year == 2018, ]

#train/test split for monthly updates
cut_off <- as.Date(Sys.time(), format = "%m/%d/%Y")
train_df2 <- neld2[neld2$dates <= cut_off, ]
test_df2 <- neld2[neld2$dates > cut_off, ]

#################################################################################
#ANALYSIS 1: INITIAL ANNUAL FORECASTS - RUN ONLY ONCE IN JANUARY OF EACH YEAR
#################################################################################
###################run super learner for all violence
#set up custom index because caretlist can't handle time slice
index = lapply(createTimeSlices(y=unique(train_df$dates), initialWindow = 1100, 
                                horizon = 1, 
                                fixedWindow = F)$train, 
               function(x) unique(train_df$dates)[x])
index = lapply(index, 
               function(x) which(train_df$dates %in% x))

indexOut = lapply(createTimeSlices(y=unique(train_df$dates), initialWindow = 1100, 
                                   horizon = 1, 
                                   fixedWindow = F)$test, 
                  function(x) unique(train_df$dates)[x])
indexOut = lapply(indexOut, 
                  function(x) which(train_df$dates %in% x))


start.time <- Sys.time()
cl <- makeCluster(8)
registerDoParallel(cl)

#algorithm setup and run
ctrl <- trainControl(summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     method = "repeatedcv",
                     number = 10,
                     repeats = 10,
                     index = index,
                     indexOut = indexOut)

set.seed(49)
#classifier fitting
#any EV, y = elecViolence1, x = -c(2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 27:29)
#gov EV, y = elecViolence2, x = -c(2:7, 20:23, 24:26, 28:29)
model_list <- caretList(elecViolence1 ~ process1 + pcgdp + logIMR + logpredict + lnpop2 + growth + lpolity2 +
                          SPI + regimetenure + year + lpolcomp + lexconst + lastelection + anticipation,
                        data = train_df,
                        methodList = c("glm", "rf", "nnet"),
                        trControl = ctrl
)

#stop parallel
stopCluster(cl)
registerDoSEQ()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken #print time taken
#inspect raw object
model_list
#compare predictions (rf, glm, nnet)
modelCor(resamples(model_list)) 
#################
#greedy optimize
#################
set.seed(49)
greedy_ensemble <- caretEnsemble(
  model_list,
  metric = "ROC",
  trControl = trainControl(
    number = 2,
    summaryFunction = twoClassSummary,
    classProbs = TRUE)
)
summary(greedy_ensemble)

#ANNUAL RISK TABLE
ens_preds <- predict(greedy_ensemble, newdata = test_df,
                     type = 'prob')
preds_df <- test_df
preds_df$prediction <- ens_preds
compare_df <- preds_df[, c("country", "dates", "elecViolence1", "prediction")]
compare_df$p_rank <- rank(compare_df$prediction)/length(compare_df$prediction)
compare_df$vio2 <- ifelse(compare_df$prediction >= .51, "vio", "peace")
compare_df$vio2 <- as.factor(compare_df$vio2)
# auc_score <- roc(compare_df$elecViolence1, compare_df$prediction, probability = TRUE, cases = "vio", control = "peace")
# auc_score
#IF test_df, create annual risk table
write.csv(compare_df, "annual_risk_table.csv")

#################################################################################
#ANALYSIS 2: MONTHLY UPDATES - RUN AT THE END OF EACH MONTH STARTING AT THE END OF JANUARY
#################################################################################
###################run super learner for all violence
#set up custom index because caretlist can't handle time slice
index2 = lapply(createTimeSlices(y=unique(train_df2$dates), initialWindow = 1100, 
                                horizon = 1, 
                                fixedWindow = F)$train, 
               function(x) unique(train_df2$dates)[x])
index2 = lapply(index2, 
               function(x) which(train_df2$dates %in% x))

indexOut2 = lapply(createTimeSlices(y=unique(train_df2$dates), initialWindow = 1100, 
                                   horizon = 1, 
                                   fixedWindow = F)$test, 
                  function(x) unique(train_df2$dates)[x])
indexOut2 = lapply(indexOut2, 
                  function(x) which(train_df2$dates %in% x))


start.time <- Sys.time()
cl <- makeCluster(8)
registerDoParallel(cl)

#algorithm setup and run
ctrl2 <- trainControl(summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     method = "repeatedcv",
                     number = 10,
                     repeats = 10,
                     index = index2,
                     indexOut = indexOut2)

set.seed(49)
#classifier fitting
#any EV, y = elecViolence1, x = -c(2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 27:29)
#gov EV, y = elecViolence2, x = -c(2:7, 20:23, 24:26, 28:29)
model_list2 <- caretList(elecViolence1 ~ process1 + pcgdp + logIMR + logpredict + lnpop2 + growth + lpolity2 +
                          SPI + regimetenure + year + lpolcomp + lexconst + lastelection + anticipation,
                        data = train_df2,
                        methodList = c("glm", "rf", "nnet"),
                        trControl = ctrl2
)

#stop parallel
stopCluster(cl)
registerDoSEQ()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken #print time taken
#inspect raw object
model_list2
#compare predictions (rf, glm, nnet)
modelCor(resamples(model_list2)) 
#################
#greedy optimize
#################
set.seed(49)
greedy_ensemble2 <- caretEnsemble(
  model_list2,
  metric = "ROC",
  trControl = trainControl(
    number = 2,
    summaryFunction = twoClassSummary,
    classProbs = TRUE)
)
summary(greedy_ensemble2)

#MONTHLY UPDATE RISK TABLE
ens_preds2 <- predict(greedy_ensemble2, newdata = test_df2,
                     type = 'prob')
preds_df2 <- test_df2
preds_df2$prediction <- ens_preds2
compare_df2 <- preds_df2[, c("country", "dates", "elecViolence1", "prediction")]
#FOR SOME REASON IT IS REVERSING THE PROBABILITIES...
compare_df2$prediction <- compare_df2$prediction
compare_df2$p_rank <- rank(compare_df2$prediction)/length(compare_df2$prediction)
compare_df2$vio2 <- ifelse(compare_df2$prediction >= .51, "vio", "peace")
compare_df2$vio2 <- as.factor(compare_df2$vio2)
# auc_score <- roc(compare_df$elecViolence1, compare_df$prediction, probability = TRUE, cases = "vio", control = "peace")
# auc_score
#IF test_df2, create update_risk_table
write.csv(compare_df2, "update_risk_table.csv")



