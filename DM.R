# IMPORT LIBRARIES
required_packages <- c("dplyr", "ggplot2", "tidyr", "readr", "ggpubr", "stringr", 
                       "ggcorrplot", "caTools", "caret", "MLmetrics", "e1071", 
                       "rpart", "rpart.plot", "class")

for (package_name in required_packages) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(package_name)
  }
}

library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(ggpubr)
library(ggplot2)
library(stringr)
library(ggcorrplot)
library(caTools)
library(caret)
library(MLmetrics)
library(e1071) # For SVM functions
library(rpart)  # For basic decision tree functions
library(rpart.plot) 
library(e1071)  # For Naive Bayes functions
library(class)

# IMPORT DATASET
data <- read.csv("ObesityDataCleaned.csv")

# DATA PREPROCESSING

# Number of Rows and Columns
dim(data)

# Display the first 6 rows of the dataset
head(data)

# Display the last 6 rows of the dataset
tail(data)

# Summary of the dataset
summary(data)

# Find out missing values
sum(is.na(data))

# change string to binary, 1 for yes and 0 for no
data$Fruits <- ifelse(data$Fruits == "Yes", 1, 0)
data$Vegetables <- ifelse(data$Vegetables == "Yes", 1, 0)
data$Meat <- ifelse(data$Meat == "Yes", 1, 0)
data$ProcFood <- ifelse(data$ProcFood == "Yes", 1, 0)
data$Sleep.8 <- ifelse(data$Sleep.8 == "YES", 1, 0)
data$Stress <- ifelse(data$Stress == "Yes", 1, 0)
data$Exercises <- ifelse(data$Exercises == "Yes", 1, 0)
data$Gender <- ifelse(data$Gender == "Male", 1, 0)


data <- data %>%
  mutate(FamilyHeart = ifelse(str_detect(FamilyHistory, regex("heart", ignore_case = TRUE)), 1, 0),
         FamilyCholesterol = ifelse(str_detect(FamilyHistory, regex("cholesterol", ignore_case = TRUE)), 1, 0)) %>%
  select(-FamilyHistory) 

# View the modified data
print(data)

# Find out outliers (This one old diagram not very detailed)
boxplot(data$BMI)
boxplot(data$Age)
boxplot(data$Weight)
boxplot(data$Height)
boxplot(data$OW)
boxplot(data$Htension)
boxplot(data$Diabetes)
boxplot(data$HC)
boxplot(data$HD)
boxplot(data$smoking)
boxplot(data$FamilyHeart)
boxplot(data$FamilyCholesterol)
boxplot(data$FriedFood)
boxplot(data$Fruits)
boxplot(data$Vegetables)
boxplot(data$Meat)
boxplot(data$ProcFood)
boxplot(data$Sleep.8)
boxplot(data$Stress)
boxplot(data$Exercises)

## Create the "images" folder if it doesn't exist
#dir.create("images", showWarnings = FALSE) 

## Define your variables
#variables <- c("BMI", "Age", "Weight", "Height", "OW", "Htension",
#"Diabetes", "HC", "HD", "smoking", "FriedFood",
#"Fruits", "Vegetables", "Meat", "ProcFood", "Sleep.8",
#"Stress", "Exercises", "FamilyHeart", "FamilyCholesterol")

## Iterate through variables, generating boxplots and saving them
#for (var in variables) {
#p <- ggplot(data, aes_string(x = var)) +
#geom_boxplot() +
#labs(title = paste("Boxplot of", var))

## Save as PNG in the "images" folder
#ggsave(filename = file.path("images", paste0(var, ".png")), plot = p) 
#}


for (col in variables) {
  cat("Unique values in '", col, "':\n")
  print(unique(data[[col]]))  
}

# Find out duplicates
sum(duplicated(data))


# remove outliers for weight
weight <- data %>% filter(Weight >110)

# remove outliers for height
height <- data %>% filter(Height < 130)

# replace the outliers with the mean
data$Weight[data$Weight > 110] <- mean(data$Weight)

# replace the outliers with the mean
data$Height[data$Height < 130] <- mean(data$Height)

# remove Occupation, EducationLevel, Race from the dataset
data <- subset(data, select = -c(Occupation, EducationLevel, Race))

# save the cleaned data
write.csv(data, file = "ObesityPreProcessedData.csv", row.names = FALSE)



cleaned_data <- read.csv('ObesityPreProcessedData.csv')

# Summary
summary(cleaned_data)

# EXPLORAORY DATA ANALYSIS

# Bar Chart

# Overweight Distribution
overweight <- table(cleaned_data$OW)
barplot(overweight, main='Overweight Distribution', xlab='Overweight (0: Not Overweight, 1: Overweight)', ylab='Frequency', col = rainbow(2))

# Heart Disease Distribution
heartDisease <- table(cleaned_data$HD)
barplot(heartDisease, main='Heart Disease Distribution', xlab='Heart Disease (0: Absence, 1: Presence)', ylab='Frequency', col = rainbow(2))

# Histogram
# Height Distribution
hist(cleaned_data$Height, main='Height Distribution', xlab='Height', ylab='Frequency', col = 'skyblue')

# Weight Distribution
hist(cleaned_data$Weight, main='Weight Distribution', xlab='Weight', ylab='Frequency', col = 'pink')

# Scatterplot

# Scatterplot for Diabetes and Heart Disease
plot(cleaned_data$Diabetes, cleaned_data$HD, main = 'Diabetes VS Heart Disease', xlab = 'Diabetes', ylab = 'Heart Disease')
cor(cleaned_data$Diabetes, cleaned_data$HD)

# Scatterplot for High Cholesterol and Heart Disease
plot(cleaned_data$HC, cleaned_data$HD, main = 'High Cholesterol VS Heart Disease', xlab = 'High Cholesterol', ylab = 'Heart Disease')

# BoxPlot

# Boxplot for BMI Distribution
boxplot(cleaned_data$BMI, main = 'BMI Distribution')

# Boxplot for Age Distribution
boxplot(cleaned_data$Age, main = 'Age Distribution')

# Heat map
cleaned_data <- subset(data, select = -`No`)
correlation_matrix <- cor(cleaned_data[2:21])

ggcorrplot(correlation_matrix, lab = TRUE, lab_size = 3)


# DATA MINING MODELLING/TECHNIQUES

# Split the data into training and testing
set.seed(123)
split_index <- sample.split(cleaned_data$HD, SplitRatio = 0.80) # 70% training, 30% testing


training_data <- subset(cleaned_data, split_index == TRUE)
testing_data  <- subset(cleaned_data, split_index == FALSE)


# Model 1: Confusion Matrix
model <- glm(HD ~ BMI + Age + Weight +  Height + OW + Htension + 
               Diabetes + HC + smoking + FriedFood + Fruits + 
               Vegetables + Meat + ProcFood + Sleep.8 + Stress + Exercises, 
             family = binomial(link = 'logit'), data = training_data)

summary(model) 

predictions <- predict(model, testing_data, type = 'response')
predictions[predictions >= 0.5] <- 1  # Threshold for positive prediction
predictions[predictions < 0.5]  <- 0


confusionMatrix(as.factor(predictions), as.factor(testing_data$HD))


# rerun model with only significant variables
model <- glm(HD ~ Htension + smoking + ProcFood,
             family = binomial(link = 'logit'), data = training_data)

summary(model)

# Predictions
predictions <- predict(model, testing_data, type = 'response')
predictions[predictions >= 0.5] <- 1  # Threshold for positive prediction
predictions[predictions < 0.5]  <- 0

# Confusion Matrix
confusionMatrix(as.factor(predictions), as.factor(testing_data$HD))

# install.packages("MLmetrics")
# Performance metrics
Precision(testing_data$HD, predictions)
Recall(testing_data$HD, predictions)
F1_Score(testing_data$HD, predictions)



# Model 2: SVM
library(e1071) # For SVM functions
library(caret) # for ConfusionMatrix

svm_model <- svm(HD ~ ., data = training_data, 
                 type = 'C-classification',  # Type of SVM for classification
                 kernel = 'radial',          # Radial basis function kernel (common choice)
                 cost = 1)                  # Regularization parameter (start with 1) 
predictions <- predict(svm_model, testing_data) 
confusionMatrix(as.factor(predictions), as.factor(testing_data$HD))

library(MLmetrics)
Precision(testing_data$HD, predictions)
Recall(testing_data$HD, predictions)
F1_Score(testing_data$HD, predictions)



# Model 3: Decision Tree
library(rpart)       # For basic decision tree functions
library(rpart.plot)  # For visualizing the tree
library(caret)       # For confusion Matrix


decision_tree <- rpart(HD ~ ., data = training_data, 
                       method = "class" ,cp = 0.05)  # Classification tree

rpart.plot(decision_tree, extra = 104)  # Visualize

predictions <- predict(decision_tree, testing_data, type = "class")
confusionMatrix(as.factor(predictions), as.factor(testing_data$HD))

library(MLmetrics)
Precision(testing_data$HD, predictions)
Recall(testing_data$HD, predictions)
F1_Score(testing_data$HD, predictions)


# Model 4: KNN
# library(class)  # For KNN functions
# library(caret)  # For confusionMatrix

# knn_model <- knn(train = training_data[, -10], test = testing_data[, -10], 
#                cl = training_data$HD, k = 5)  # 5 nearest neighbors

# confusionMatrix(knn_model, as.factor(testing_data$HD))
# library(MLmetrics)
# Precision(testing_data$HD, predictions)
# Recall(testing_data$HD, predictions)
# F1_Score(testing_data$HD, predictions)


# Model 5: Naive Bayes
# library(e1071)  # For Naive Bayes functions

# naive_bayes <- naiveBayes(HD ~ ., data = training_data)

# predictions <- predict(naive_bayes, testing_data)
# confusionMatrix(predictions, as.factor(testing_data$HD))

# library(MLmetrics)
# Precision(testing_data$HD, predictions)
# Recall(testing_data$HD, predictions)
# F1_Score(testing_data$HD, predictions)


