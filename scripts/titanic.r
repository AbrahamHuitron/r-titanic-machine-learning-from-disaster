# installing and loading packages required
install.packages("pacman")
library("pacman")

p_load("dplyr", "randomForest", "caret")

# setting working director to allow for file access
setwd("~/Projects/R/titanic-machine-learning-from-disaster")

# importing data file
train_data <- read.csv("data/final/train.csv")
test_data <- read.csv("data/final/test.csv")

# displaying first 6 rows of the data files
head(train_data)
head(test_data)

# women who survived
women = filter(train_data, Sex == "female")
women_survived = filter(train_data, Sex == "female" & Survived == 1)
rate_women = nrow(women_survived)/nrow(women)

cat("% of women who survived:", rate_women)

# men who survived
men = filter(train_data, Sex == "male")
men_survived = filter(train_data, Sex == "male" & Survived == 1)
rate_men = nrow(men_survived)/nrow(men)

cat("% of men who survived:", rate_men)

# converting and confirming variable datatypes for use in random forest
# https://bookdown.org/gmli64/do_a_data_science_project_in_10_days/random-forest-with-key-predictors.html
y <- as.factor(train_data$Survived)

features <- c("Pclass", "Sex", "SibSp", "Parch")
X <- train_data[features]
X_test <- test_data[features]

X$Pclass <- as.factor(X$Pclass)
X$Sex <- as.factor(X$Sex)
X$SibSp <- as.factor(X$SibSp)

sapply(X, class)

X_test$Pclass <- as.factor(X_test$Pclass)
X_test$Sex <- as.factor(X_test$Sex)
X_test$SibSp <- as.factor(X_test$SibSp)

sapply(X_test, class)

# creating first random forest model
rf_model1 <- randomForest(y ~ Pclass + Sex + SibSp + Parch
                          ,data=X, importance=TRUE)

# checking model accuracy
rf_model1

# making a prediction using our validation data set
# the original train variable is a list so it needs to be converted to
# a data frame
rf_prediction1 <- predict(rf_model1, X)

#storing confusion matrix and model statistics
conMat1 <- confusionMatrix(rf_prediction1, y)

# kaggle submission variables
rf_prediction_kaggle <- predict(rf_model1, X_test)
conMat2 <- confusionMatrix(rf_prediction_kaggle, y)

submit <- data.frame(PassengerId = test_data$PassengerId, Survived = rf_prediction_kaggle)
write.csv(submit, file = "./output/rf_model1_result.csv", row.names = FALSE)