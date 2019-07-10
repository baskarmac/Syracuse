#Data Mining Project
#Baskar Dakshinamoorthy
#Lauren Foltz


#Notes on Data
#m = original data
#m2 = veiltype column removed (only one level)
#armDF = veiltype and stalkroot removed for Association Rule Mining

###################################################################################################
# Prep Work
###################################################################################################

## Clean up:Detach previously loaded packages, if desired.
#search() #to see what's currently loaded
#lapply(paste('package:',names(sessionInfo()$otherPkgs),sep=""),detach,character.only=TRUE,unload=TRUE)


## Check current working directory and change if needed. Then comment out.
getwd()
#setwd("C:/Users/laure/OneDrive/Documents/Syracuse MS/03 Fall Classes/02 Sun 707 Data Mining/R Files for IST707/DM Project")
#setwd("/Users/baskar/Documents/ml/Syracuse/IST-707/IST_707_Project")

#==================================================================================================
# Text Mining
#==================================================================================================
#install.packages("tm")
library(tm) #text mining, for reading corpus
## Loading required package: NLP
library(stringr)
library(wordcloud)
## Loading required package: RColorBrewer
library(slam)
#library(quanteda) #masks as.DocumentTermMatrix and stopwords in tm
library(SnowballC)
#library(proxy) # masks as.matrix from base package and Matrix package
library(stringi)
library(Matrix)
#The following object is masked from package:tidyr:expand
library(tidytext) # convert DTM to DF
library(textmineR)
#library(igraph) #masks knn from class package
#library(lsa) #latent semantic analysis


## Put desired text doc in the project folder, in a folder called "corpus"

## Read in text document; Corpus should appear in the Environment under Data with length of 1
Corpus <- Corpus(DirSource("corpus")) 
## view Corpus; should come up in console as "Simple Corpus"
(Corpus)
## View summary; make sure title is as expected
summary(Corpus)
## Store length as ndocs; it should appear under Values
(ndocs<-length(Corpus)) 

## Clean the Corpus and create Document Term Matrix; dtm should appear under Data
dtm <- DocumentTermMatrix(Corpus,
                          control = list(
                            stopwords = TRUE, 
                            wordLengths=c(3, 15),
                            removePunctuation = T,
                            removeNumbers = T,
                            tolower=T,
                            #stemming = T,
                            remove_separators = T
                              ))

## Perform Checks, either by viewing or with inspect
#(dtm)
inspect(dtm) # shows number of documents and terms
## We expect sparcity to be 0 because there is only one document

## Convert what we just built into a matrix; dtm_M should appear under Data
dtm_M <- as.matrix(dtm)
## Check it
(dtm_M[1,1:10])

## Look at word frequencies; WordFreq
WordFreq <- colSums(dtm_M) #get column sums and store as WordFreq (should appear in Values)
(length(WordFreq)) #get length in console
ord <- order(WordFreq) #put them in order, and store as ord (should appear under values)
(WordFreq[head(ord)]) #least frequent will appear in console
(WordFreq[tail(ord)]) #most frequent will appear in console

## Row Sums gives a number for each document, good for normalizing by hand.
#(Row_Sum_Per_doc <- rowSums(dtm_M))

## Convert un-normalized Matrix to a DataFrame; mush_DF should appear under Data
mush_DF <- as.data.frame(dtm_M)  
## View portion of Structure
str(mush_DF [1,1:10])

## Check how many rows are in the DF; it should be the same as the number of text documents
(nrow(mush_DF)) 

## Create a word cloud using a matrix
wordcloud(colnames(dtm_M), dtm_M[1, ], max.words = 50,colors=brewer.pal(8,"Dark2"))

## If desired, select a word to look up. Will get the freqency in each document
(mush_DF$beds)
(mush_DF$manure)
(mush_DF$spawn)
(mush_DF$cellar)
(mush_DF$loam)
(mush_DF$temperature)

(mush_DF$compost)
(mush_DF$straw)
(mush_DF$agar)
(mush_DF$species)

#note: can rerun with other books, just need to swap out the corpus

###################################################################################################
# Load Libraries
###################################################################################################

# Load Libraries
library(ggplot2)
library(plyr)
## The following object is masked from package:NLP :annotate
#install.packages("tidyverse")
library(tidyverse)
library(e1071)
#install.packages("mlr")
library(mlr)
###The following object is masked from package:e1071: impute
#install.packages("caret",
#                 repos = "http://cran.r-project.org", 
#                 dependencies = TRUE)
library(caret)
###The following object is masked from package:mlr: train
###The following object is masked from package:purrr:lift
#install.packages("naivebayes")
library(naivebayes)
library(mclust)
###The following object is masked from package:purrr: map
library(cluster)
#install.packages("rpart")
library(rpart)
#install.packages('rattle')
library(rattle)
#install.packages('rpart.plot')
library(rpart.plot)
#install.packages('RColorBrewer')
#library(RColorBrewer) #already loaded in textmining portion
#install.packages("Cairo")
library(Cairo)
# install.packages("corrplot")
library(corrplot)



##Load Files

#place "mushrooms.csv" in the R project folder
#read the file into R using the read.csv function
#use header=TRUE to let R know that headers are present
#Use "na.strings = "NA" so R will replace spaces/blanks with "NA"
#Note: R will convert dashes to dots (cap-shape will be cap.shape)
filename="mushrooms.csv"
m<- read.csv(filename, header = TRUE, na.strings = "NA",stringsAsFactors = TRUE)
#m now appears in the Environment under Data as 8124 obs. of 23 variables

## Look at the data, then comment out
#View(m)

## Check for missing values
Total <-sum(is.na(m))
cat("The number of missing values in Mushroom data is ", Total )
#The number of missing values in Mushroom data is  0

#Explore the data

#look at a table
(table(m$class))
(table(m$class,m$habitat))
#use a loop to create all of the tables at once, then review the data
for(i in 1:ncol(m)){
  print(table(m[i]))
}


#The tables show that there are 2480 instances of "?" in column L "stalk-root"

#Notes on things you can do are below
#(colnames(m2)) # to get column names
#(head(m2)) # to see the first 6 rows
#(m2) # to see all the data

## Which variables contain important information?
#veil.type is not valuable. There is only one level.
#All other variables may have value.
#Class is important; it is our label.

#use "str" to check the data types.
str(m)
#all are factors.

###################################################################################################
# Update Variable Names
###################################################################################################

#look at barplot, update variable, check barplot again to make sure correct
par(mfrow=c(2,1)) #set plot parameters for easiy comparison
barplot(table(m$class))
m$class<-recode(m$class, e = "Edible", p = "Poisonous")
barplot(table(m$class))

barplot(table(m$cap.shape))
m$cap.shape<-recode(m$cap.shape, b = "Bell", c = "Conical" , x = "Convex" , f = "Flat" , k = "Knobbed" , s = "Sunken")
barplot(table(m$cap.shape))

barplot(table(m$cap.surface))
m$cap.surface<-recode(m$cap.surface, f = "Fibrous", g = "Grooves" , y = "Scaly" , s = "Smooth" )
barplot(table(m$cap.surface))

barplot(table(m$cap.color))
m$cap.color<-recode(m$cap.color, n = "Brown", b= "Buff" , c = "Cinnamon" , g = "Gray", r = "Green", p = "Pink" , u = "Purple" , e = "Red" , w = "White" ,y = "Yellow")
barplot(table(m$cap.color))

barplot(table(m$bruises))
m$bruises<-recode(m$bruises, t = "Bruises", f= "No")
barplot(table(m$bruises))

barplot(table(m$odor))
m$odor <-recode(m$odor, a = "Almond", l= "Anise" , c = "Creosote" , y = "Fishy", f = "Foul", m = "Musty" , n = "None" , p = "Pungent" , s = "Spicy")
barplot(table(m$odor))

barplot(table(m$gill.attachment))
m$gill.attachment<-recode(m$gill.attachment, a = "Attached", f = "Free")
barplot(table(m$gill.attachment))

barplot(table(m$gill.spacing))
m$gill.spacing<-recode(m$gill.spacing, c = "Close", w = "Crowded")
barplot(table(m$gill.spacing))

barplot(table(m$gill.size))
m$gill.size<-recode(m$gill.size, b = "Broad", n = "Narrow")
barplot(table(m$gill.size))

barplot(table(m$gill.color))
m$gill.color<-recode(m$gill.color, k = "Black", n = "Brown", b= "Buff" , h = "Chocolate" , g = "Gray", r = "Green", o = "Orange", p = "Pink" , u = "Purple" , e = "Red" , w = "White" ,y = "Yellow")
barplot(table(m$gill.color))

barplot(table(m$stalk.shape))
m$stalk.shape<-recode(m$stalk.shape, e = "Enlarging", t = "Tapering")
barplot(table(m$stalk.shape))

barplot(table(m$stalk.root))
m$stalk.root<-recode(m$stalk.root, b = "Bulbous", c = "Club", e = "Equal" ,r = "Rooted")
barplot(table(m$stalk.root))

barplot(table(m$stalk.surface.above.ring))
m$stalk.surface.above.ring<-recode(m$stalk.surface.above.ring, f= "fibrous", y = "Scaly", k= "Silky" , s = "Smooth")
barplot(table(m$stalk.surface.above.ring))

barplot(table(m$stalk.surface.below.ring))
m$stalk.surface.below.ring<-recode(m$stalk.surface.below.ring, f= "fibrous", y = "Scaly", k= "Silky" , s = "Smooth")
barplot(table(m$stalk.surface.below.ring))

barplot(table(m$stalk.color.above.ring))
m$stalk.color.above.ring<-recode(m$stalk.color.above.ring, n = "Brown", b= "Buff" , c = "Cinnamon" , g = "Gray", o = "Orange", p = "Pink" , e = "Red" , w = "White" ,y = "Yellow")
barplot(table(m$stalk.color.above.ring))

barplot(table(m$stalk.color.below.ring))
m$stalk.color.below.ring<-recode(m$stalk.color.below.ring , n = "Brown", b= "Buff" , c = "Cinnamon" , g = "Gray", o = "Orange", p = "Pink" , e = "Red" , w = "White" ,y = "Yellow")
barplot(table(m$stalk.color.below.ring))

barplot(table(m$veil.type))
m$veil.type<-recode(m$veil.type , p = "Partial")
barplot(table(m$veil.type))

barplot(table(m$veil.color))
m$veil.color<-recode(m$veil.color , n = "Brown", o = "Orange", w = "White" ,y = "Yellow")
barplot(table(m$veil.color))

barplot(table(m$ring.number))
m$ring.number<-recode(m$ring.number , n = "None", o = "One", t = "Two")
barplot(table(m$ring.number))

barplot(table(m$ring.type))
m$ring.type<-recode(m$ring.type , e= "Evanescent" , f = "Flaring" , l = "Large", n = "None", p = "Pendant" )
barplot(table(m$ring.type))

barplot(table(m$spore.print.color))
m$spore.print.color<-recode(m$spore.print.color , k = "Black", n = "Brown", b= "Buff" ,h = "Chocolate" , r = "Green", o = "Orange" , u = "Purple" , w = "White" ,y = "Yellow")
barplot(table(m$spore.print.color))

barplot(table(m$population))
m$population<-recode(m$population , a = "Abundant", c= "Clustered" , n = "Numerous" , s = "Scattered", v= "Several", y = "Solitary")
barplot(table(m$population))

barplot(table(m$habitat))
m$habitat<-recode(m$habitat , g = "Grasses", l= "Leaves" , m = "Meadows" , p = "Paths", u= "Urban", w = "Waste", d = "Woods")
barplot(table(m$habitat))


###################################################################################################
# Barchart visualization for paper
###################################################################################################

#Set A
par(mfrow=c(2,2))
barplot(table(m$class), 
        col='green',
        xlab='Class',ylab='Count',
        main='Class')
barplot(table(m$cap.shape),
        col='firebrick1',
        xlab='Cap Shape',ylab='Count',
        main='Cap Shape')
barplot(table(m$cap.surface),
        col='dodgerblue3',
        xlab='Cap Surface',ylab='Count',
        main='Cap Surface')
barplot(table(m$cap.color),
        col='yellow',
        xlab='Cap Color',ylab='Count',
        main='Cap Color')

#Set B
par(mfrow=c(2,2))
barplot(table(m$bruises), 
        col='green',
        xlab='Bruises',ylab='Count',
        main='Bruises')
barplot(table(m$odor),
        col='firebrick1',
        xlab='Odor',ylab='Count',
        main='Odor')
barplot(table(m$gill.attachment),
        col='dodgerblue3',
        xlab='Gill Attachment',ylab='Count',
        main='Gill Attachment')
barplot(table(m$gill.spacing),
        col='yellow',
        xlab='Gill Spacing',ylab='Count',
        main='Gill Spacing')

#Set C
par(mfrow=c(2,2))
barplot(table(m$gill.size),
        col='green',
        xlab='Gill Size',ylab='Count',
        main='Gill Size')
barplot(table(m$gill.color),
        col='firebrick1',
        xlab='Gill Color',ylab='Count',
        main='Gill Color')
barplot(table(m$stalk.shape),
        col='dodgerblue3',
        xlab='Stalk Shape',ylab='Count',
        main='Stalk Shape')
barplot(table(m$stalk.root),
        col='yellow',
        xlab='Stalk Root',ylab='Count',
        main='Stalk Root')

#Set D
par(mfrow=c(2,2))
barplot(table(m$stalk.surface.above.ring),
        col='green',
        xlab='Stalk Surface',ylab='Count',
        main='Stalk Surface')
barplot(table(m$stalk.surface.below.ring),
        col='firebrick1',
        xlab='Stalk Surface Below Ring',ylab='Count',
        main='Stalk Surface Below Ring')
barplot(table(m$stalk.color.above.ring),
        col='dodgerblue3',
        xlab='Stalk Color Above Ring',ylab='Count',
        main='Stalk Color Above Ring')
barplot(table(m$stalk.color.below.ring),
        col='yellow',
        xlab='Stalk Color Below Ring',ylab='Count',
        main='Stalk Color Below Ring')

#Set E
par(mfrow=c(2,2))
barplot(table(m$veil.color),
        col='green',
        xlab='Veil Color',ylab='Count',
        main='Veil Color')
barplot(table(m$ring.number),
        col='firebrick1',
        xlab='Ring Number',ylab='Count',
        main='Ring Number')
barplot(table(m$ring.type),
        col='dodgerblue3',
        xlab='Ring Type',ylab='Count',
        main='Ring Type')
barplot(table(m$spore.print.color),
        col='yellow',
        xlab='Spore Print Color',ylab='Count',
        main='Spore Print Color')

#Set F
par(mfrow=c(2,2))
barplot(table(m$spore.print.color),
        col='green',
        xlab='Spore Print Color',ylab='Count',
        main='Spore Print Color')
barplot(table(m$population),
        col='firebrick1',
        xlab='Population',ylab='Count',
        main='Population')
barplot(table(m$habitat),
        col='dodgerblue3',
        xlab='Habitat',ylab='Count',
        main='Habitat')


## Reset parameters
par(mfrow=c(1,1))

#==================================================================================================
# Creating Training Set and Testing set for models Decision Tree and Random Forest Classification
#==================================================================================================

#Make a copy of the data frame and removed veil.type as it has only one value
m2<-m[,-c(17)]
str(m2)
#Create Training and Test Dataset using Sample method.Took 70% for Training and 30% for testing data
n = nrow(m2)
n
trainIndex = sample(1:n, size = round(0.7*n), replace=FALSE)
#Create Training Set
mush_train = m2[trainIndex ,]
str(mush_train)
dim(mush_train)
#Create Testing set
mush_test = m2[-trainIndex ,]
dim(mush_test)
#Remove the Decision Class from the Testing set
mush_test_nolabels<-mush_test[-c(1)]
dim(mush_test_nolabels)
#Created a Data Frame with just the training labels
mush_train_nolabels<-mush_train[-c(1)]
dim(mush_train_nolabels)
str(mush_test_nolabels)
#Create a Lable DF for Class variable
TestClassLabels<-mush_test$class
length(TestClassLabels)
TrainClassLabels<-mush_train$class
length(TrainClassLabels)

#==================================================================================================
# Decision Tree Classification Modelling
#==================================================================================================

#Running the Model to find out what variables are important
set.seed(123)
#Create a model with split as gini
model <- rpart( class ~ ., data=mush_train,method="class",parms = list(split = "gini"))
summary(model)

#Create the Decision Tree and save as Jpeg
jpeg("DecisionTree_Mushroom_Sample1.jpg")
fancyRpartPlot(model)
rpart.plot(model,extr=101)
dev.off()
plotcp(model)
printcp(model)


#Do the prediction
(mush_test_nolabels)
predicted=predict(model,mush_test_nolabels,type="class")
length(predicted)
(Results<-data.frame(predicted=predicted,Actual=TestClassLabels))
(table(Results))

#Calculate Accuracy using the ConfusionMatrix- Accuracy comes in at 99.6%
confusionMatrix(predicted,TestClassLabels)

#I tried different combination of Variables against Decision Class 
model1 <- rpart( class ~ odor+spore.print.color+gill.color+stalk.surface.above.ring+stalk.color.below.ring+ring.type,
                 data=mush_train,method="class",control = rpart.control(cp = 0, maxdepth = 8,minsplit = 10))
summary(model1)
plotcp(model1)
printcp(model1)
fancyRpartPlot(model1)
predicted=predict(model1,mush_test_nolabels,type="class")
(Results<-data.frame(predicted=predicted,Actual=TestClassLabels))
(table(Results))
confusionMatrix(predicted,TestClassLabels)

#class ~ cap.color+cap.shape+habitat -Accuracy Comes in at 78%. We can do this for all other combinations but looks like Odor and Spore.print.color is variables of Importance.
model2 <- rpart( class ~ cap.color+cap.shape+habitat, data=mush_train,method="class",minsplit = 2, minbucket = 1)
summary(model2)
fancyRpartPlot(model2)
predicted=predict(model2,mush_test_nolabels,type="class")
(Results<-data.frame(predicted=predicted,Actual=TestClassLabels))
(table(Results))
confusionMatrix(predicted,TestClassLabels)
#model <- rpart( class ~ cap.shape, data=mush_train,method="class",minsplit = 2, minbucket = 1)
#model <- rpart( class ~ habitat, data=mush_train,method="class",minsplit = 2, minbucket = 1)


#Another Sample method to validate our results from previous Model
(every7_indexes<-seq(1,nrow(m2),7))
mush_Df_sampletest=m2[every7_indexes, ]
mush_Df_sampletrain=m2[-every7_indexes, ]
str(mush_Df_sampletrain)
dim(mush_Df_sampletrain)
mush_test_labels<-mush_Df_sampletest$class
mush_Df_sampletest1<-mush_Df_sampletest[-c(1)]

head(mush_test_labels)
TestClassLabels1<-mush_test_labels
dim(mush_Df_sampletest1)
model_new <- rpart( class ~ ., data=mush_Df_sampletrain,method="class")
#model_new <- rpart( class ~ habitat, data=mush_Df_sampletrain,method="class",minsplit = 2, minbucket = 1)
summary(model)

jpeg("DecisionTree_Mushroom_Sample2.jpg")
fancyRpartPlot(model_new)
dev.off()



#Do the Prediction using Sample 2
predicted=predict(model_new,mush_Df_sampletest1,type="class")
(Results<-data.frame(predicted=predicted,Actual=TestClassLabels1))
(table(Results))

#Calculate Accuracy using the ConfusionMatrix- Accuracy comes in at 99%
confusionMatrix(predicted,TestClassLabels1)

length(TestClassLabels1)



#==================================================================================================
# Random Forest to predict the importance of Variables and avoid overfitting
#==================================================================================================

library(randomForest)
#The following object is masked from package:rattle:importance
#The following object is masked from package:dplyr:combine
#The following object is masked from package:ggplot:margin

#Default model with ntree=500
set.seed(100)
head(mush_train)
rf_model1 = randomForest(class ~ .,
                         data = mush_train,importance=TRUE)
plot(rf_model1,main='Error vs Trees')

#Evaluate the performance of the model
predict_rf <- predict(rf_model1, newdata = mush_test_nolabels)
confusionMatrix(predict_rf, TestClassLabels)

#Plotting the Variables according to its Importance
print(rf_model1)

varImpPlot(rf_model1,
           sort = T,
           n.var = 10,
           main = "Variable Importance")

varImpPlot(rf_model1,type=2)
#Create a dataframe with Variable Importance
var.imp = data.frame(importance(rf_model1,type=2))

# List Variables according to its importance derived from Gini MeanDecrease
var.imp$Variables = row.names(var.imp)
print(var.imp[order(var.imp$MeanDecreaseGini,decreasing = T),])



#Evaluate the performance of the model1
predict_rf <- predict(rf_model1, newdata = mush_test_nolabels)
confusionMatrix(predict_rf, TestClassLabels)

# Creating a RF model with Different mtry and ntrees
#This will take 15 minutes to run as we are tuning the model with different mtry values
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
grid_rf <- expand.grid(.mtry = c(2,4,8,16))
##Kappa was used to select the optimal model using the largest value.
##The final value used for the model was mtry = 4.
rf_model2 <- train(class ~ ., data = mush_train, method = "rf", metric = "Kappa", trControl = ctrl, tuneGrid = grid_rf)
rf_model2
plot(rf_model2,main='Error vs Trees')

print(rf_model2)


#Do the prediction
rfPredict=predict(rf_model2,mush_test_nolabels)
(Results<-data.frame(predicted=rfPredict,Actual=TestClassLabels))
(table(Results))

#Calcualte Accuracy using the ConfusionMatrix- Accuracy comes in at 99%
#Random Forest Accuracy comes in at 100% compared to Decision Tree Accuracy of 99.38% 
confusionMatrix(rfPredict,TestClassLabels)


#==================================================================================================
# K-Nearest Neighbours
#==================================================================================================

library(class)
library(gmodels)
#install.packages("tictoc")
library(tictoc)

set.seed(101)
tic()

##knn_model <- knn(train=mush_train, test=mushroom,cl=TrainClassLabels,k=11) # create KNN model

#Training the Knn model                                                               
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3,classProbs = TRUE)
#Training the model -> Will run for 302.954 sec 
knn_fit <- train(class~., data=mush_train, method = "knn",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)
toc()
knn_fit
#Accuracy was used to select the optimal model using the largest value.
###The final value used for the model was k = 7.
summary(knn_fit)
plot(knn_fit)
#Evaluate the performance of the model
knnPredict<-predict(knn_fit,mush_test_nolabels)

#How Accurately our model is working?
confusionMatrix(knnPredict,TestClassLabels)



#==================================================================================================
#Support Vector Machines
#==================================================================================================

#Create a Model with default Values and do the prediction
svm_model <- svm(class~., data=mush_train, type='C-classification', kernel='radial') # create svm model
# we set the kernel to radial as this data set does not a have a linear plane that can be drawn
summary(svm_model)

test_svm <-predict(svm_model,mush_test_nolabels) # predicting with the new SVM model


mean(test_svm==TestClassLabels)  # percentage of testset predicted correctly by svm
(Results<-data.frame(predicted=test_svm,Actual=TestClassLabels))
table(Results)  # confusion matrix of the predictions of the svm and the test data

# perform a SVM tune to get the optimal Cost and Gamma Values-10-fold cross validation 
svm_tune <- tune(svm, class~., data = mush_train,                                            #takes a while to run
                 kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
#- best parameters:
###cost gamma
####1   0.5
print(svm_tune)

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3,classProbs = TRUE)

set.seed(101)
#Support Vector Machines with Linear Kernel and 10 Fold Cross-Validation
model_fit <- train(class~., data=mush_train, method = "svmLinear",
                   trControl=trctrl,
                   preProcess = c("center", "scale"),
                   tuneLength = 10)

print(model_fit);
# Classify from our reserved test set.
testing_set_predict = predict(model_fit, newdata = mush_test_nolabels); 
# Verifying our model from the classifications.
table(testing_set_predict, TestClassLabels);
confusionMatrix(testing_set_predict,TestClassLabels)

#Run the SVM Model one more time with tuned Gamma and Cost Values

svm_model1 <- svm(class~., data=mush_train, type='C-classification', kernel='radial',cost=1,gamma=0.5)
# Classify from our reserved test set.
testing_set_predict1 = predict(svm_model1, newdata = mush_test_nolabels); 
# Verifying our model from the classifications.
table(testing_set_predict1, TestClassLabels);
confusionMatrix(testing_set_predict1,TestClassLabels)

#==================================================================================================
# CLustering using K-Means,K-mode and Rock
#==================================================================================================

#install.packages("cluster")
library(cluster)
##install.packages("factoextra")
library(factoextra) ## for DBSCAN
#install.packages("klaR")
library(klaR)
#install.packages("cba")
library(cba)

#Reading the dataset again
filename="mushrooms.csv"
mushroomDf<- read.csv(filename, header = TRUE, na.strings = "NA")
any(is.na(mushroomDf))
#Removing the Class variable and Veil.type from the dataset
mushroomDf.torun <- subset(mushroomDf, select = -c(class, veil.type))

#Clustering using k-means by one-hot encoding

#This is basically creating dummy variables for each value of the category, for all the variables.
mushroomDf.torun.ohe <- model.matrix(~.-1, data=mushroomDf.torun)

str(mushroomDf.torun.ohe)

set.seed(20) #for reproducibility
#nstart = 50, indicates R will run 50 different random starting assignments and selects the lowest within cluster variation
result.kmean = kmeans(mushroomDf.torun.ohe, 2, nstart = 50, iter.max = 15) 
print(result.kmean)
result.kmean3 <- kmeans(mushroomDf.torun.ohe, centers = 3, nstart = 25)
result.kmean4 <- kmeans(mushroomDf.torun.ohe, centers = 4, nstart = 25)
result.kmean5 <- kmeans(mushroomDf.torun.ohe, centers = 5, nstart = 25)


#Purity of clustering is a simple measure of the accuracy, which is between 0 and 1. 0 indicates poor clustering, and 1 indicates perfect clustering
#Purity of Cluster with K=2
result.kmean.mm <- table(mushroomDf$class, result.kmean$cluster)
purity.kmean <- sum(apply(result.kmean.mm, 2, max)) / nrow(mushroomDf.torun)
purity.kmean
#Purity of Cluster with K=3
result.kmean3.mm <- table(mushroomDf$class, result.kmean3$cluster)
purity.kmean3 <- sum(apply(result.kmean3.mm, 2, max)) / nrow(mushroomDf.torun)
purity.kmean3
#Purity of Cluster with K=4
result.kmean4.mm <- table(mushroomDf$class, result.kmean4$cluster)
purity.kmean4 <- sum(apply(result.kmean4.mm, 2, max)) / nrow(mushroomDf.torun)
purity.kmean4
#Purity of Cluster with K=5
result.kmean5.mm <- table(mushroomDf$class, result.kmean5$cluster)
purity.kmean5 <- sum(apply(result.kmean5.mm, 2, max)) / nrow(mushroomDf.torun)
purity.kmean5
# Creating plots to compare between different K-Values
p1 <- fviz_cluster(result.kmean, geom = "point", data = mushroomDf.torun.ohe) + ggtitle("k = 2")
p2 <- fviz_cluster(result.kmean3, geom = "point",  data = mushroomDf.torun.ohe) + ggtitle("k = 3")
p3 <- fviz_cluster(result.kmean4, geom = "point",  data = mushroomDf.torun.ohe) + ggtitle("k = 4")
p4 <- fviz_cluster(result.kmean5, geom = "point",  data = mushroomDf.torun.ohe) + ggtitle("k = 5")

library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)

################################################
#Clustering using K-mode with different K-Values
################################################
set.seed(20) #for reproducibility
#nstart = 50, indicates R will run 50 different random starting assignments and selects the lowest within cluster variation
result.kmode <- kmodes(mushroomDf.torun.ohe, 2, iter.max = 50, weighted = FALSE)
print(result.kmode)

result.kmode.mm <- table(mushroomDf$class, result.kmode$cluster)
result.kmode.mm
purity.kmode <- sum(apply(result.kmode.mm, 2, max)) / nrow(mushroomDf.torun)
purity.kmode
result.kmode3 <- kmodes(mushroomDf.torun.ohe,3,iter.max = 50, weighted = FALSE)
print(result.kmode3)
result.kmode3.mm <- table(mushroomDf$class, result.kmode3$cluster)
result.kmode3.mm
purity.kmode3 <- sum(apply(result.kmode3.mm, 2, max)) / nrow(mushroomDf.torun)
purity.kmode3
result.kmode4 <- kmodes(mushroomDf.torun.ohe,4,iter.max = 50, weighted = FALSE)
print(result.kmode4)
result.kmode4.mm <- table(mushroomDf$class, result.kmode4$cluster)
purity.kmode4 <- sum(apply(result.kmode4.mm, 2, max)) / nrow(mushroomDf.torun)
purity.kmode4
result.kmode5 <- kmodes(mushroomDf.torun.ohe,5,iter.max = 50, weighted = FALSE)
print(result.kmode5)
result.kmode5.mm <- table(mushroomDf$class, result.kmode5$cluster)
purity.kmode5 <- sum(apply(result.kmode5.mm, 2, max)) / nrow(mushroomDf.torun)
purity.kmode5

# plots to compare
par(mfrow=c(2,2))
clusplot(mushroomDf.torun, result.kmode$cluster, color=TRUE, shade=TRUE, labels=2, lines=0,main='K-Value 2') 
clusplot(mushroomDf.torun, result.kmode3$cluster, color=TRUE, shade=TRUE, labels=2, lines=0,main='K-Value 3')
clusplot(mushroomDf.torun, result.kmode4$cluster, color=TRUE, shade=TRUE, labels=2, lines=0,main='K-Value 4')
clusplot(mushroomDf.torun, result.kmode5$cluster, color=TRUE, shade=TRUE, labels=2, lines=0,main='K-Value 5')


#Clustering using Rock
mushroom.torun.binary <- as.dummy(mushroomDf.torun)
result.rock <-rockCluster(mushroom.torun.binary, n=5, theta=0.8)
result.rock.mm<-table(mushroomDf$class, result.rock$cl)
purity.rock <- sum(apply(result.rock.mm, 2, max)) / nrow(mushroomDf.torun)
purity.rock


# Compute hierarchical clustering
mushroomDf.torun2 <- subset(mushroomDf, select = c(odor,stalk.color.below.ring,stalk.color.above.ring,habitat,gill.size,gill.color,population,ring.number))
head(mushroomDf.torun2)
#Clustering using k-means by one-hot encoding

#This is basically creating dummy variables for each value of the category, for all the variables.
mushroomDf.torun.ohe2 <- model.matrix(~.-1, data=mushroomDf.torun2)
mushroomDf.torun.ohe3<- scale(mushroomDf.torun.ohe2)
head(mushroomDf.torun.ohe3)
# Dissimilarity matrix
d <- dist(mushroomDf.torun.ohe2, method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "complete" )


# Cut tree into 4 groups
sub_grp <- cutree(hc1, k = 5)
table(sub_grp)
fviz_cluster(list(data = mushroomDf.torun.ohe2, cluster = sub_grp))

# Enhanced k-means clustering
res.km <- eclust(mushroomDf.torun.ohe2, "kmeans", nstart = 25,nboot=5)   #takes 5 minutes to run
# Gap statistic plot
fviz_gap_stat(res.km$gap_stat)
# Silhouette plot
fviz_silhouette(res.km)

## list of cluster assignments

fviz_cluster(res.km, df,  geom = "point", 
             ellipse= FALSE, show.clust.cent = FALSE,
             palette = "jco", ggtheme = theme_classic())
                     


## DBSCAN Density-based clustering with columns 
#install.packages("dbscan")
library(dbscan)
df <- mushroomDf.torun.ohe

# create a vector of epsilon values

epsilon_values <- c(1.8, 0.5, 0.4)

# plot the distribution of distances

kNNdistplot(df, k = 5)

# plot lines at epsilon values

for (e in epsilon_values) {
  abline(h = e, col = "red")
}

for (e in epsilon_values) {
  db_clusters <- dbscan(df, eps=e, minPts=4)
  title <- paste("Plot for epsilon = ", e)
  g <- fviz_cluster(db_clusters, df, ellipse = TRUE, geom = "point",
                    main = title)
  print(g)
}


#==================================================================================================
# Naive Bayes
#==================================================================================================

#create another set without stalk.root or veil.type

#create training set
str(mush_train)
mush_train2<-mush_train[,-c(12)]
str(mush_train2)
dim(mush_train2)

#create testing set
str(mush_test)
mush_test2<-mush_test[,-c(12)]
str(mush_test2)
dim(mush_train2)

#create training with no labels
mush_train2_no_labels<-mush_train2[,-c(1)]
str(mush_train2_no_labels)
dim(mush_train2_no_labels)

#create testing with no labels
mush_test2_no_labels<-mush_test2[,-c(1)]
str(mush_test2_no_labels)
dim(mush_train2_no_labels) 


#NB Model #1 (using Naive Bayes package)

nb <- naive_bayes( class ~. , data= mush_train2 , laplace=1, na.action = na.pass )# create model with train set
nb_prediction <- predict ( nb ,  mush_test2 ) #create prediction with test set
(cm = table(nb_prediction , TestClassLabels)) #get confusion matrix
model_accuracy = sum(diag(cm))/sum(cm) #calculate accuracy
a<-(model_accuracy_p<- paste(round((model_accuracy)*100,digits=2),"%",sep="")) #convert to a percent
cat("The accuracy of the Naive Bayes model is ", model_accuracy_p )

#Visualize Naive Bayes
#This gives a separate vis per variable
plot(nb, legend.box = TRUE) 


#look at information gain with Entropy
#install.packages("CORElearn")
library(CORElearn)
Method.CORElearn <- CORElearn::attrEval(mush_train2$class ~ ., data=mush_train2,  estimator = "GainRatio")
max(Method.CORElearn)
which.max(Method.CORElearn)


#NB Model #2 (using e1071 package)
library(e1071)
NB_e1071<-naiveBayes (class~., data=mush_train2, na.action = na.pass)
NB_e1071_Pred <- predict(NB_e1071, mush_test2_no_labels) # takes a while
(cm = table(NB_e1071_Pred, TestClassLabels)) #get confusion matrix
model_accuracy = sum(diag(cm))/sum(cm) #calculate accuracy
model_accuracy_p<- paste(round((model_accuracy)*100,digits=2),"%",sep="") #convert to a percent
cat("The accuracy of the Naive Bayes model made with package e1071 is ", model_accuracy_p )

#plot(NB_e1071, ylab = "Density", main = "Naive Bayes Plot")
# Error in xy.coords(x, y, xlabel, ylabel, log) : 
#  'x' is a list, but does not have components 'x' and 'y'


#==================================================================================================
# Association Rule Mining
#==================================================================================================

#load libraries
#install.packages("arules")
library("arules")
#The following object is masked from package:dplyr: recode
#The following object is masked from package:tm:inspect
#The following objects are masked from package:base: abbreviate, write
#install.packages("arulesViz")
library("arulesViz")

#Adjust Data

#make a copy of data frame with veil.type removed. Call it armDF
armDF <- m2
str(armDF)

#remove column for stalk.root because it has "?" in some cells
which( colnames(armDF)=="stalk.root" ) #12 means stalk.root is column 12
armDF <-armDF[-c(12)]
which( colnames(armDF)=="stalk.root" ) #0 means stalk.root is no longer present

#make a copy and remove the label, which is the first column
armDF_unlabeled <-armDF[-c(1)]
str(armDF_unlabeled)


################
# Arm Model 1  #
################

#explore unlabeled data
rules<-arules::apriori(armDF_unlabeled,parameter = list(supp=0.50, conf = 0.95,minlen=2))
rules <- rules[!is.redundant(rules)] #remove redundant
options(digits=3)
#produced 76 rules

plot(rules, main = "Association Rules Model #1 (76 rules)") #add title

#sort by conf & inspect
rules_conf<-sort (rules, decreasing = TRUE, by='confidence')
inspect(rules_conf[1:10])

#top 10 rules:
#lhs                                                       rhs                    support confidence lift count
#[1]  {stalk.color.below.ring=White}                         => {gill.attachment=Free} 0.540   1          1.03 4384 
#[2]  {stalk.color.below.ring=White}                         => {veil.color=White}     0.540   1          1.03 4384 
#[3]  {stalk.color.above.ring=White}                         => {gill.attachment=Free} 0.549   1          1.03 4464 
#[4]  {stalk.color.above.ring=White}                         => {veil.color=White}     0.549   1          1.03 4464 
#[5]  {stalk.shape=Tapering}                                 => {ring.number=One}      0.567   1          1.08 4608 
#[6]  {stalk.shape=Tapering}                                 => {gill.attachment=Free} 0.567   1          1.03 4608 
#[7]  {stalk.shape=Tapering}                                 => {veil.color=White}     0.567   1          1.03 4608 
#[8]  {gill.attachment=Free,stalk.surface.below.ring=Smooth} => {veil.color=White}     0.584   1          1.03 4744 
#[9]  {stalk.surface.below.ring=Smooth,veil.color=White}     => {gill.attachment=Free} 0.584   1          1.03 4744 
#[10] {gill.attachment=Free,stalk.surface.above.ring=Smooth} => {veil.color=White}     0.613   1          1.03 4984 

#sort by support
rules_supp <- sort(rules, decreasing = TRUE, by="supp")
inspect(rules_supp[1:10])

#top 10 rules:
#lhs                                          rhs                    support confidence lift  count
#[1]  {gill.attachment=Free}                    => {veil.color=White}     0.973   0.999      1.024 7906 
#[2]  {veil.color=White}                        => {gill.attachment=Free} 0.973   0.998      1.024 7906 
#[3]  {ring.number=One}                         => {gill.attachment=Free} 0.898   0.974      1.000 7296 
#[4]  {ring.number=One}                         => {veil.color=White}     0.897   0.973      0.998 7288 
#[5]  {veil.color=White,ring.number=One}        => {gill.attachment=Free} 0.897   1.000      1.027 7288 
#[6]  {gill.spacing=Close}                      => {veil.color=White}     0.815   0.972      0.996 6620 
#[7]  {gill.spacing=Close}                      => {gill.attachment=Free} 0.813   0.969      0.995 6602 
#[8]  {gill.attachment=Free,gill.spacing=Close} => {veil.color=White}     0.813   1.000      1.025 6602 
#[9]  {gill.attachment=Free,gill.spacing=Close} => {ring.number=One}      0.772   0.950      1.031 6272 
#[10] {gill.size=Broad}                         => {veil.color=White}     0.667   0.966      0.990 5420 


#sort by lift
rules_lift <- sort(rules, decreasing = TRUE,by="lift")
inspect(rules_lift[1:10])

#top 10 rules:
#lhs                                                   rhs                    support confidence lift count
#[1]  {stalk.shape=Tapering}                             => {ring.number=One}      0.567   1.00       1.08 4608 
#[2]  {gill.attachment=Free,gill.spacing=Close}          => {ring.number=One}      0.772   0.95       1.03 6272 
#[3]  {stalk.color.below.ring=White}                     => {gill.attachment=Free} 0.540   1.00       1.03 4384 
#[4]  {stalk.color.above.ring=White}                     => {gill.attachment=Free} 0.549   1.00       1.03 4464 
#[5]  {stalk.shape=Tapering}                             => {gill.attachment=Free} 0.567   1.00       1.03 4608 
#[6]  {stalk.surface.below.ring=Smooth,veil.color=White} => {gill.attachment=Free} 0.584   1.00       1.03 4744 
#[7]  {stalk.surface.above.ring=Smooth,veil.color=White} => {gill.attachment=Free} 0.613   1.00       1.03 4984 
#[8]  {veil.color=White,ring.number=One}                 => {gill.attachment=Free} 0.897   1.00       1.03 7288 
#[9]  {stalk.color.below.ring=White}                     => {veil.color=White}     0.540   1.00       1.03 4384 
#[10] {stalk.color.above.ring=White}                     => {veil.color=White}     0.549   1.00       1.03 4464 



################
# Arm Model 2  #
################

#what happens if I reduce confidence to 0.90 and supp to 0.40?
rules<-arules::apriori(armDF_unlabeled,parameter = list(supp=0.40, conf = 0.90,minlen=2))
rules <- rules[!is.redundant(rules)] #remove redundant
options(digits=3)
#produced 283 rules

plot(rules, main = "Association Rules Model #2 (283 rules)") #add title
#we got more rules, and the lift range increased

#sort by conf & inspect
rules_conf<-sort (rules, decreasing = TRUE, by='confidence')
rules <- rules[!is.redundant(rules)] #remove redundant
inspect(rules_conf[1:10])

#top 10 rules:
#lhs                               rhs                    support confidence lift count
#[1]  {bruises=Bruises}              => {gill.attachment=Free} 0.416   1          1.03 3376 
#[2]  {bruises=Bruises}              => {veil.color=White}     0.416   1          1.03 3376 
#[3]  {stalk.color.below.ring=White} => {gill.attachment=Free} 0.540   1          1.03 4384 
#[4]  {stalk.color.below.ring=White} => {veil.color=White}     0.540   1          1.03 4384 
#[5]  {stalk.color.above.ring=White} => {gill.attachment=Free} 0.549   1          1.03 4464 
#[6]  {stalk.color.above.ring=White} => {veil.color=White}     0.549   1          1.03 4464 
#[7]  {stalk.shape=Tapering}         => {ring.number=One}      0.567   1          1.08 4608 
#[8]  {stalk.shape=Tapering}         => {gill.attachment=Free} 0.567   1          1.03 4608 
#[9]  {stalk.shape=Tapering}         => {veil.color=White}     0.567   1          1.03 4608 
#[10] {odor=None,veil.color=White}   => {gill.attachment=Free} 0.410   1          1.03 3328 


#sort by support
rules_supp <- sort(rules, decreasing = TRUE, by="supp")
inspect(rules_supp[1:10])

#top 10 rules:
#lhs                                          rhs                    support confidence lift  count
#[1]  {gill.attachment=Free}                    => {veil.color=White}     0.973   0.999      1.024 7906 
#[2]  {veil.color=White}                        => {gill.attachment=Free} 0.973   0.998      1.024 7906 
#[3]  {ring.number=One}                         => {gill.attachment=Free} 0.898   0.974      1.000 7296 
#[4]  {gill.attachment=Free}                    => {ring.number=One}      0.898   0.922      1.000 7296 
#[5]  {ring.number=One}                         => {veil.color=White}     0.897   0.973      0.998 7288 
#[6]  {veil.color=White}                        => {ring.number=One}      0.897   0.920      0.998 7288 
#[7]  {veil.color=White,ring.number=One}        => {gill.attachment=Free} 0.897   1.000      1.027 7288 
#[8]  {gill.spacing=Close}                      => {veil.color=White}     0.815   0.972      0.996 6620 
#[9]  {gill.spacing=Close}                      => {gill.attachment=Free} 0.813   0.969      0.995 6602 
#[10] {gill.attachment=Free,gill.spacing=Close} => {veil.color=White}     0.813   1.000      1.025 6602 



#sort by lift
rules_lift <- sort(rules, decreasing = TRUE,by="lift")
inspect(rules_lift[1:10])

#top 10 rules:
#lhs                                                    rhs                               support confidence lift count
#[1]  {ring.number=One,ring.type=Pendant}                 => {stalk.surface.above.ring=Smooth} 0.420   0.960      1.51 3416 
#[2]  {stalk.surface.below.ring=Smooth,ring.type=Pendant} => {stalk.surface.above.ring=Smooth} 0.410   0.959      1.50 3328 
#[3]  {gill.spacing=Close,ring.type=Pendant}              => {stalk.surface.above.ring=Smooth} 0.409   0.954      1.50 3320 
#[4]  {stalk.surface.above.ring=Smooth,ring.type=Pendant} => {stalk.surface.below.ring=Smooth} 0.410   0.908      1.49 3328 
#[5]  {ring.type=Pendant}                                 => {stalk.surface.above.ring=Smooth} 0.451   0.923      1.45 3664 
#[6]  {odor=None}                                         => {gill.size=Broad}                 0.405   0.932      1.35 3288 
#[7]  {bruises=Bruises}                                   => {gill.spacing=Close}              0.403   0.969      1.16 3272 
#[8]  {population=Several}                                => {gill.spacing=Close}              0.474   0.952      1.14 3848 
#[9]  {ring.number=One,ring.type=Pendant}                 => {gill.spacing=Close}              0.414   0.944      1.13 3360 
#[10] {stalk.shape=Tapering}                              => {ring.number=One}                 0.567   1.000      1.08 4608 


################
# Arm Model 3  #
################

#Now let's use labeled data and set the class as the rhs, to find out what's most associated with poison/edible
#supp of .40 gave no rules, so reduce it to .30
rules<-arules::apriori(data = armDF, parameter = list(supp=0.30, conf = 0.90, minlen=2),
                       appearance = list(default="lhs",rhs="class=Poisonous"),
                       control = list(verbose = F))
options(digits=3)
rules <- rules[!is.redundant(rules)] #remove redundant rules
rules
#produced 12 rules

plot(rules, main = "Association Rules Model #3 (12 rules)") #add title
#lift looks strong

#sort by conf & inspect
rules_conf<-sort (rules, decreasing = TRUE, by='confidence')
inspect(rules_conf[1:10])

#sort by support
rules_supp <- sort(rules, decreasing = TRUE, by="supp")
inspect(rules_supp[1:10])

#sort by lift
rules_lift <- sort(rules, decreasing = TRUE,by="lift")
inspect(rules_lift[1:10])

#This produced some pretty complicated rules that centered around the same 6 attributes
#     lhs                                                                                                                  rhs               support confidence lift count
#  {bruises=No, gill.attachment=Free,gill.spacing=Close,                    population=Several}                          => {class=Poisonous} 0.302   0.972      2.02 2456 
#  {bruises=No,                      gill.spacing=Close, veil.color=White,  population=Several}                          => {class=Poisonous} 0.302   0.972      2.02 2456 
#  {bruises=No,                      gill.spacing=Close,                    population=Several}                          => {class=Poisonous} 0.302   0.936      1.94 2456 
#  {bruises=No, gill.attachment=Free,gill.spacing=Close,                                         ring.number=One}        => {class=Poisonous} 0.388   0.956      1.98 3152 
#  {bruises=No,                      gill.spacing=Close, veil.color=White,                       ring.number=One}        => {class=Poisonous} 0.388   0.956      1.98 3152 
#  {bruises=No, gill.attachment=Free,gill.spacing=Close}                                                                 => {class=Poisonous} 0.390   0.952      1.97 3170 
#  {bruises=No,                      gill.spacing=Close, veil.color=White}                                               => {class=Poisonous} 0.392   0.952      1.98 3188 
#  {bruises=No, gill.attachment=Free,                                      population=Several}                           => {class=Poisonous} 0.308   0.954      1.98 2504 
#  {bruises=No,                                         veil.color=White,  population=Several}                           => {class=Poisonous} 0.308   0.954      1.98 2504 


#PLOT POISONOUS
plot(rules, method="graph", engine="interactive")



################
# Arm Model 4  #
################

#let's reduce rule length to get simpler rules
#maxlen = 2 produced no rules, so go with 3
rules<-arules::apriori(data = armDF, parameter = list(supp=0.30, conf = 0.90, minlen=2, maxlen=3),
                       appearance = list(default="lhs",rhs="class=Poisonous"),
                       control = list(verbose = F))
options(digits=3)
rules <- rules[!is.redundant(rules)] #remove redundant
rules
#produced 2 rules

plot(rules, main = "Association Rules Model #4 (2 rules)") #add title
#lift looks strong
inspect(rules)


#gill.spacing had higher support, but population had stronger confidence.
#I don't think "population several" is very useful though,
#so let's go with bruises = No and Gill Spacing = Close as the most important variables.

#lhs                                rhs               support confidence lift count
#[1] {bruises=No,population=Several} => {class=Poisonous} 0.308   0.921      1.91 2504 
#[2] {bruises=No,gill.spacing=Close} => {class=Poisonous} 0.392   0.901      1.87 3188 




################
# Arm Model 5  #
################

#Now let's set the rhs to edible
rules<-arules::apriori(data = armDF, parameter = list(supp=0.30, conf = 0.90, minlen=2, maxlen =3),
                       appearance = list(default="lhs",rhs="class=Edible"),
                       control = list(verbose = F))
options(digits=3)
rules <- rules[!is.redundant(rules)] #let's remove redundant rules this time
rules
plot(rules, main = "Association Rules Model #5 (9 rules)") #add title
#lift looks strong
inspect(rules)

#We have 9 rules. Odor= None seems to have a strong association.
# lhs                                                   rhs           support confidence lift count
# {odor=None}                                       => {class=Edible} 0.419   0.966      1.86 3408 
# {odor=None,stalk.shape=Tapering}                  => {class=Edible} 0.307   1.000      1.93 2496 
# {odor=None,stalk.surface.below.ring=Smooth}       => {class=Edible} 0.344   0.972      1.88 2792 
# {odor=None,stalk.surface.above.ring=Smooth}       => {class=Edible} 0.350   0.973      1.88 2840 
# {odor=None,gill.size=Broad}                       => {class=Edible} 0.396   0.978      1.89 3216 
# {odor=None,ring.number=One}                       => {class=Edible} 0.355   0.984      1.90 2880 
# {odor=None,veil.color=White}                      => {class=Edible} 0.396   0.966      1.87 3216 
# {gill.size=Broad,stalk.surface.below.ring=Smooth} => {class=Edible} 0.392   0.936      1.81 3184 
# {gill.size=Broad,stalk.surface.above.ring=Smooth} => {class=Edible} 0.416   0.940      1.81 3376 




################
# Arm Model 6  #
################

#Make a slightly larger model of Edible for an interactive plot
rules<-arules::apriori(data = armDF, parameter = list(supp=0.30, conf = 0.90, minlen=2),
                       appearance = list(default="lhs",rhs="class=Edible"),
                       control = list(verbose = F))
rules <- rules[!is.redundant(rules)] #remove redundant
rules #17 rules
plot(rules, main = "Association Rules Model #6 (17 rules)") #add title

#PLOT EDIBLE
plot(rules, method="graph", engine="interactive")



################
# Arm Model 7  #
################
#Per Instructions, set the LHS 
#I'll try setting LHS to Poisonous
rules<-arules::apriori(data = armDF, parameter = list(supp=0.01, conf = 0.60, minlen=2),
                       appearance = list(lhs = "class=Poisonous",default = "rhs"),
                       control = list(verbose = F))
options(digits=3)
rules <- rules[!is.redundant(rules)] #remove redundant rules
rules
#produced 6 rules

plot(rules, main = "Association Rules Model #7 (6 rules)") #add title

#sort by conf & inspect
rules_conf<-sort (rules, decreasing = TRUE, by='confidence')
inspect(rules_conf)

#lhs                  rhs                    support confidence lift count
#[1] {class=Poisonous} => {veil.color=White}     0.481   0.998      1.02 3908 
#[2] {class=Poisonous} => {gill.attachment=Free} 0.480   0.995      1.02 3898 
#[3] {class=Poisonous} => {ring.number=One}      0.469   0.972      1.06 3808 
#[4] {class=Poisonous} => {gill.spacing=Close}   0.468   0.971      1.16 3804 
#[5] {class=Poisonous} => {bruises=No}           0.405   0.841      1.44 3292 
#[6] {class=Poisonous} => {population=Several}   0.351   0.727      1.46 2848 

#sort by support
rules_supp <- sort(rules, decreasing = TRUE, by="supp")
inspect(rules_supp)

#sort by lift
rules_lift <- sort(rules, decreasing = TRUE,by="lift")
inspect(rules_lift)




################
# Arm Model 8  #
################
#Now try setting LHS to Edible
rules<-arules::apriori(data = armDF, parameter = list(supp=0.01, conf = 0.60, minlen=2),
                       appearance = list(lhs = "class=Edible",default = "rhs"),
                       control = list(verbose = F))
options(digits=3)
rules <- rules[!is.redundant(rules)] #remove redundant rules
rules
#produced 13 rules

plot(rules, main = "Association Rules Model #8 (13 rules)") #add title


#sort by conf & inspect
rules_conf<-sort (rules, decreasing = TRUE, by='confidence')
inspect(rules_conf[1:10])
#lhs               rhs                               support confidence lift  count
#[1]  {class=Edible} => {gill.attachment=Free}            0.494   0.954      0.980 4016 
#[2]  {class=Edible} => {veil.color=White}                0.494   0.954      0.978 4016 
#[3]  {class=Edible} => {gill.size=Broad}                 0.483   0.932      1.349 3920 
#[4]  {class=Edible} => {ring.number=One}                 0.453   0.875      0.949 3680 
#[5]  {class=Edible} => {stalk.surface.above.ring=Smooth} 0.448   0.865      1.358 3640 
#[6]  {class=Edible} => {odor=None}                       0.419   0.810      1.865 3408 
#[7]  {class=Edible} => {stalk.surface.below.ring=Smooth} 0.419   0.808      1.330 3400 
#[8]  {class=Edible} => {ring.type=Pendant}               0.388   0.749      1.534 3152 
#[9]  {class=Edible} => {gill.spacing=Close}              0.370   0.715      0.853 3008 
#[10] {class=Edible} => {bruises=Bruises}                 0.339   0.654      1.574 2752 


#sort by support
rules_supp <- sort(rules, decreasing = TRUE, by="supp")
inspect(rules_supp[1:10])
#lhs               rhs                               support confidence lift  count
#[1]  {class=Edible} => {gill.attachment=Free}            0.494   0.954      0.980 4016 
#[2]  {class=Edible} => {veil.color=White}                0.494   0.954      0.978 4016 
#[3]  {class=Edible} => {gill.size=Broad}                 0.483   0.932      1.349 3920 
#[4]  {class=Edible} => {ring.number=One}                 0.453   0.875      0.949 3680 
#[5]  {class=Edible} => {stalk.surface.above.ring=Smooth} 0.448   0.865      1.358 3640 
#[6]  {class=Edible} => {odor=None}                       0.419   0.810      1.865 3408 
#[7]  {class=Edible} => {stalk.surface.below.ring=Smooth} 0.419   0.808      1.330 3400 
#[8]  {class=Edible} => {ring.type=Pendant}               0.388   0.749      1.534 3152 
#[9]  {class=Edible} => {gill.spacing=Close}              0.370   0.715      0.853 3008 
#[10] {class=Edible} => {bruises=Bruises}                 0.339   0.654      1.574 2752 

#sort by lift
rules_lift <- sort(rules, decreasing = TRUE,by="lift")
inspect(rules_lift[1:10])
#lhs               rhs                               support confidence lift count
#[1]  {class=Edible} => {odor=None}                       0.419   0.810      1.86 3408 
#[2]  {class=Edible} => {bruises=Bruises}                 0.339   0.654      1.57 2752 
#[3]  {class=Edible} => {ring.type=Pendant}               0.388   0.749      1.53 3152 
#[4]  {class=Edible} => {stalk.surface.above.ring=Smooth} 0.448   0.865      1.36 3640 
#[5]  {class=Edible} => {gill.size=Broad}                 0.483   0.932      1.35 3920 
#[6]  {class=Edible} => {stalk.surface.below.ring=Smooth} 0.419   0.808      1.33 3400 
#[7]  {class=Edible} => {stalk.color.below.ring=White}    0.333   0.643      1.19 2704 
#[8]  {class=Edible} => {stalk.color.above.ring=White}    0.339   0.654      1.19 2752 
#[9]  {class=Edible} => {stalk.shape=Tapering}            0.319   0.616      1.09 2592 
#[10] {class=Edible} => {gill.attachment=Free}            0.494   0.954      0.98 4016 






##################
# End of project 
##################




