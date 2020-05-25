install.packages("glmnet")
install.packages("leaps")
install.packages('pls')
install.packages('randomForest')
install.packages('GGally')
install.packages('dplyr')
library(plyr)
library(dplyr)
library(MASS)
library(leaps)
library(glmnet)
library(pls)
library(randomForest)
library(lattice)
library(ggplot2)
library(GGally)
library(reshape2)
library(reshape)
library(boot)


#load the data
baseball=read.csv("~/strikeouts.csv",header=T, sep = ',')
head(baseball)
colnames(baseball)
summary(baseball)
dim(baseball)


#load the plate discipline data from fangraphs
leaderboard=read.csv("~/Plate_Discipline_Data.csv",header=T, sep = ',',stringsAsFactors=FALSE)
leaderboard=na.omit(leaderboard)
head(leaderboard)
fangraphs_id = leaderboard[,c(12)]
leaderboard_subset = leaderboard[,c(3:11)]
#change the data from percentages to decimals
leaderboard1 = apply(leaderboard_subset,2, function(x){
  as.numeric(sub("%", "", x, fixed=TRUE))/100})
head(leaderboard1)

#bind the id column to the leaderbard dataframe
leaderboard_final = cbind(leaderboard1[,c(1,2,4,5,7:9)], fangraphs_id)
leaderboard_final= as.data.frame(leaderboard_final)
head(leaderboard_final)
dim(leaderboard_final)

#join the original data table with the average fastball velocity table and swing strike percent table - join by fangraphs id column
final_strikeouts = inner_join(baseball, leaderboard_final, by = 'fangraphs_id')
head(final_strikeouts)
dim(final_strikeouts)
final_strikeouts= na.omit(final_strikeouts)
typeof(final_strikeouts)

#reorder the columns
final_strikeouts = final_strikeouts[,c(1:16,19:25,17,18)]
head(final_strikeouts)
colnames(final_strikeouts)

#make dataset of only data needed to build models
data_final = final_strikeouts[c(4:24)]
head(data_final)

#multivariate scatter matrix - correlations between predictor variables
ggpairs(data_final[c(1:20)])#shows distribution of each variable and correlation with all other variables
cor(data_final[,c(1:20)])#correlation matrix


#linear regression for each individual variable
data_final_plot = melt(data_final, id.vars='X2ndHalfK.')
ggplot(data_final_plot) +
  geom_point(aes(value,X2ndHalfK., colour=variable)) + geom_smooth(aes(value,X2ndHalfK., colour=variable), method=lm, se=FALSE) +
  facet_wrap(~variable, scales="free_x") +
  labs(y = "2nd Half K Percentage")


#split into training and test set
train_index = sample(1:nrow(data_final), 0.8 * nrow(data_final))
test_index = setdiff(1:nrow(data_final), train_index)

# Build X_train, y_train, X_test, y_test
X_train = data_final[train_index,]
y_train = data_final[train_index, "X2ndHalfK."]

X_test = data_final[test_index,]
y_test = data_final[test_index, "X2ndHalfK."]

#normalize training and test data
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
X_train_norm = as.data.frame(lapply(X_train[1:20], min_max_norm))
X_test_norm = as.data.frame(lapply(X_test[1:20], min_max_norm))
X_train_norm = cbind(X_train_norm, X_train[,21])
X_test_norm = cbind(X_test_norm, X_test[,21])

colnames(X_train_norm)[which(names(X_train_norm) == "X_train[, 21]")] = "X2ndHalfK."
colnames(X_test_norm)[which(names(X_test_norm) == "X_test[, 21]")] = "X2ndHalfK."


#regression model on training data using all the predictors
regression.model = lm(X2ndHalfK.~., data = X_train)
summary(regression.model)

#best subset selection for variables in linear regression model
train.model = regsubsets(X2ndHalfK.~.,data = X_train)
summary(train.model)
train.summary = summary(train.model)
train.summary$adjr2
train.summary$rss
train.summary$cp


#10-fold cross validation approach
predict.regsubsets =function (object ,newdata ,id ,...){
   form=as.formula (object$call [[2]])
   mat=model.matrix (form ,newdata )
   coefi =coef(object ,id=id)
   xvars =names (coefi )
   mat[,xvars ]%*% coefi
   }

k=10
set.seed(10)#make results reproducible
folds = sample(1:k,nrow(X_train),replace = TRUE)
cv.errors = matrix(NA,k,20, dimnames = list(NULL,paste(1:20)))

for(j in 1:k){
  best.fit = regsubsets(X2ndHalfK.~.,data=X_train[folds!=j,],nvmax=20)
  for(i in 1:20){
    pred = predict(best.fit,X_train[folds==j,],id=i)
    cv.errors[j,i]=mean((X_train$X2ndHalfK.[folds==j]-pred)^2)
  }
}

#find mean test MSE
mean.cv.errors = apply(cv.errors,2,mean)
which.min(mean.cv.errors)#selects 3 variable model
#perform best subset selection on full training set in order to obtain the 3 variable model
train.best= regsubsets(X2ndHalfK.~., data = X_train, nvmax = 20)
coef(train.best,3)
#predict on test set
regsubset_predict = 0.3767157 + .5321670*X_test$K. + .1825437*X_test$LD. + -.3629673*X_test$Z.Contact
#compute MSE
mean((regsubset_predict-y_test)^2)#.00173

#format training sets into matrices instead of data frames
X_train1 = model.matrix(X2ndHalfK.~.,X_train)[,-1]
X_test1 = model.matrix(X2ndHalfK.~.,X_test)[,-1]
X_train1_norm = model.matrix(X2ndHalfK.~.,X_train_norm)[,-1]
X_test1_norm = model.matrix(X2ndHalfK.~.,X_test_norm)[,-1]

#lasso - use normalized data
#first exploratory lasso model with untrained lambda 
lasso.model = glmnet(X_train1_norm, y_train, alpha = 1)
plot(lasso.model)

#find the best lambda using cross-validation on training set(use default setting of 10 folds, error as MSE)
set.seed(1)
cv.lasso=cv.glmnet(X_train1_norm,y_train,alpha=1)
plot(cv.lasso)
bestlam = cv.lasso$lambda.min

#find MSE on test set using optimized lambda parameter
lasso.pred = predict(lasso.model,s=bestlam, newx = X_test1_norm)
mean((lasso.pred-y_test)^2)#.00266

#find coefficients of final lasso model using the optimized lambda on the full dataset
out=glmnet(x,y,alpha = 1,lambda = bestlam)
lasso.coef = predict(out,type = "coefficients", s=bestlam)
lasso.coef

#partial least squares regression
set.seed(1)
pls.fit = plsr(X2ndHalfK.~., data = X_train, scale=TRUE,validation="CV")
summary(pls.fit)

#plot for which number partial least squares directions give the best MSE
validationplot(pls.fit,val.type = "MSEP")

#apply final partial least squares model to the test set
pls.pred = predict(pls.fit,X_test,ncomp=2)
mean((pls.pred-y_test)^2)#.00168


#random forest
set.seed(15)
forest.fit =  randomForest(X2ndHalfK.~., data=X_train, mtry=7, importance=TRUE)

#predict random forest on test data
predict_forest = predict(forest.fit,newdata=X_test)
mean((y_test -predict_forest)^2)#.00215

#find importance of each variable
importance(forest.fit)

#make predictions for whole dataset using preferred model
full.model = regsubsets(X2ndHalfK.~., data = data_final, nvmax = 20)
coef(full.model,3)
summary = summary(full.model)
#r-squared for the model
summary$rsq[3]
#model predictions
regression_model_predict = 0.3429920 + .5399349*data_final$K. + .2233910*data_final$LD. + -.3351884*data_final$Z.Contact
#add predictions to dataset
data_with_predictions = cbind(final_strikeouts,regression_model_predict)
#data frame with just name, 2nd Half K, and prediction
name_predictions = data_with_predictions[,c(1,24,26)]
#add column for difference between prediction and actual 2nd half K percentage
difference = regression_model_predict - data_final$X2ndHalfK.
name_predictions = cbind(name_predictions,difference)
#MSE
mean((regression_model_predict-data_final$X2ndHalfK.)^2)
plot1 = ggplot(name_predictions)
plot2 = plot1 + geom_point(aes(X2ndHalfK.,regression_model_predict)) + geom_abline(intercept = 0, slope = 1) + ggtitle("Actual vs Predicted 2nd Half K Percentage") + labs(y= "Model Prediction",x = "Actual 2nd Half K%")
plot2

#average distance in absolute value between predictiona nd actual 2nd half K percentage
difference1 = mean(abs(difference))

write.csv(name_predictions, file = "Model Predictions.csv")

