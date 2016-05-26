# Find the mistakes in exercising with a Dumbbell Biceps Curl
Thorsten Gomann  
May, 26th 2016  

Six young health participants did exercises with a Dumbbell Biceps Curl. They did exercises in a correct way, but also in four incorrect ways. Four sets of sensors capture data of the movements. This analysis tries to use these data to fit a prediction model to classify, given new data, if the exercise is correct or which kind of mistake was made.

Data provided by:
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz49MAF0cz4


## Data processing
First step is to load relevant libraries and to get the data from the given source.

```r
library(caret);library(randomForest)

if (!file.exists("pml-training.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                  "pml-training.csv")
}
df <- read.csv("pml-training.csv")
```
The next step is to separate a test dataset by splitting into 60 % for training and 40 % for testing:

```r
set.seed(123)
inTrain <- createDataPartition(df$classe, p=0.6, list=FALSE)

training <- df[inTrain, ]
testing <- df[-inTrain, ]
```
After practicing some exploratory steps like str and summary (to long output for this documentation) I guess it's a good idea to replace NA's by zero. 

The first six columns contain descriptive data, not relevant for the classification.
Also the calculated values (containing Div by 0 Error) I decided to remove. They are just a result of other information (highly correlated) provided in this dataset.


```r
training[is.na(training)] <- 0
exclude <- grep("^skew|^kurtosis|^max_yaw|^min_yaw|^amplitude_yaw", names(df))
training <- training[, c(-1:-6, -exclude)]
```

### Model selection and fitting the model
The captured data are continuous information (no categorical values) and the result should be a classification. To predict 20 individual datasets, the model should be highly accurate (Calculating a probability of predicting 20 individuals correctly the accuracy should be above 95%).
Calculation time and interpretability is no matter, so I decided to use randomforest for model creation.

Within the fitting of the model I defined 10 cross validations repeated 10 times and setting a seed makes it reproducible.

```r
set.seed(123)
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
fit <- randomForest(classe~., data=training, importance=TRUE, proximity=TRUE, trControl = fitControl)
```

### Test the model
Before predicting I have to repeat the changes from the training set to the testing set. So there's only one step: I convert also NA's to zero. After using the model to to create the predictions, the confusion matrix shows the highly accurate model. Only a few datasets are predicted incorrectly with a total accuracy rate of 99.81 % with only small variation within a 95 % confidence interval. Concluding only a small amount of out of sample errors.

```r
testing[is.na(testing)] <- 0
pred <- predict(fit, testing)
confusionMatrix(data = as.character(pred), as.character(testing$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230    2    0    0    0
##          B    2 1516    3    0    0
##          C    0    0 1365    7    0
##          D    0    0    0 1279    1
##          E    0    0    0    0 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9981          
##                  95% CI : (0.9968, 0.9989)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9976          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9987   0.9978   0.9946   0.9993
## Specificity            0.9996   0.9992   0.9989   0.9998   1.0000
## Pos Pred Value         0.9991   0.9967   0.9949   0.9992   1.0000
## Neg Pred Value         0.9996   0.9997   0.9995   0.9989   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1932   0.1740   0.1630   0.1837
## Detection Prevalence   0.2845   0.1939   0.1749   0.1631   0.1837
## Balanced Accuracy      0.9994   0.9989   0.9984   0.9972   0.9997
```

```r
confusionMatrix(data = as.character(predict(fit, training)), as.character(training$classe))$overall[1]
```

```
##  Accuracy 
## 0.9984715
```
Compared to the in sample error the accuracy is nearly identical. (Depending on the seed even a better out of sample error vs. in sample error "occurred")

## Summary
Machine learning algorithms - especially the method random forest - are able to train a model using the captured data to give a highly accurate prediction.
