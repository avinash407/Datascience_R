getwd()
setwd('C:\\Users\\avina\\Downloads\\SVR\\SVR')
dataset = read.csv('Position_Salaries.csv')
dataset
dataset <-dataset[2:3]
dataset
lin_reg=lm(formula = Salary ~., data =dataset)
summary(lin_reg)

#Fitting polynomial regression model
dataset$Level2<-dataset$Level^2
dataset$Level3<-dataset$Level^3
dataset$Level4<-dataset$Level^4
poly_reg=lm(formula = Salary ~., data =dataset)
summary(poly_reg)

library(ggplot2)

ggplot() + geom_point(aes(x=dataset$Level,y=dataset$Salary), color='red') +
  geom_line(aes(x=dataset$Level,y=predict(lin_reg,newdata=dataset)), color ='blue')

ggplot() + geom_point(aes(x=dataset$Level,y=dataset$Salary), color='red') +
  geom_line(aes(x=dataset$Level,y=predict(poly_reg,newdata=dataset)), color ='blue')



# decision tree regression

library(rpart)

regressor = rpart()

help(rpart)
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]
regressor = rpart(formula= Salary ~., data=dataset)

y_pred = predict(regressor, data.frame(Level =6.5))
y_pred

library(ggplot2)

ggplot() + geom_point(aes(x=dataset$Level,y=dataset$Salary), color='red') +
  geom_line(aes(x=dataset$Level,y=predict(regressor,newdata=dataset)), color ='blue')
# for the above gg plot since we got a straight line which does not tell any information
#or prediction, it is of no use, so we need to use rpart control to split the data

regressor = rpart(formula= Salary ~., data=dataset, control = rpart.control(minsplit = 1))

ggplot() + geom_point(aes(x=dataset$Level,y=dataset$Salary), color='red') +
  geom_line(aes(x=dataset$Level,y=predict(regressor,newdata=dataset)), color ='blue')
# but still this plot does not give a accurate prediction because of the continous nature
#so we need to use high resolution plot for this.

X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot()+geom_point(aes(x=dataset$Level, y=dataset$Salary),colour ='red')+
  geom_line(aes(x=X_grid,y=predict(regressor,newdata = data.frame(Level =X_grid))))


# Random forest regression

library(randomForest)
set.seed(1234)
regressor = randomForest(x=dataset[1], y= dataset$Salary, ntree = 10)
y_pred = predict(regressor, data.frame(Level=6.5))
y_pred
library(ggplot2)

X_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot()+geom_point(aes(x=dataset$Level, y=dataset$Salary),colour ='red')+
  geom_line(aes(x=X_grid,y=predict(regressor,newdata = data.frame(Level =X_grid))))
 # now increase the number of trees to get the accurate prediction

#Logistic regression
setwd('D:\\R and python lecture')
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]

library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set= subset(dataset,split=TRUE)
test_set=subset(dataset,split=FALSE)
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2] = scale(test_set[,1:2])

classifier = glm(formula =Purchased ~., family = binomial,data=training_set)

prob_pred = predict(classifier, type='response', newdata = test_set[-3])
prob_pred

y_pred = ifelse(prob_pred>0.5,1,0)
y_pred

cm = table(test_set[,3], y_pred)
cm

#Apriori algorithm

setwd('D:\\R and python lecture')
library(arules)

dataset <- read.csv('Market_Basket_Optimisation.csv')
dataset = read.transactions('Market_Basket_Optimisation.csv', sep=',',rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset,topN=100)  #This is the top products purchased by customers in the store

#Training apriori on the dataset

rules = apriori(data=dataset,parameter=list(support= 0.003, confidence =0.8 ))
#the support value is derived by: 3 times a product is purchased a day multiplied by 7
#which makes the whole week transactions and then dividing it by 7501 which are the total
#transactions.., i.e.., 3*7/7501

#After executing this step we got no rules written in it, so we need to decrease the
#confidence level of our model

rules = apriori(data=dataset,parameter=list(support= 0.003, confidence =0.4 ))

#Visualizing the results

inspect(sort(rules, by = 'lift')[1:10])

#Again reducing the confidence by half

rules = apriori(data=dataset,parameter=list(support= 0.003, confidence =0.2 ))
inspect(sort(rules, by = 'lift')[1:10])


#Eclat model

dataset <- read.csv('Market_Basket_Optimisation.csv')
dataset = read.transactions('Market_Basket_Optimisation.csv', sep=',',rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset,topN=100)

rules = eclat(data=dataset,parameter=list(support= 0.004, minlen=2 ))

inspect(sort(rules, by = 'support')[1:10])


#Natural language processing
dataset_original = read.delim('Restaurant_Reviews.tsv',quote='',stringsAsFactors = FALSE)
dataset_original

#cleaning the texts
install.packages('tm')
install.packages('SnowballC') #This is used for stopwords
library('tm')
library('SnowballC')
corpus = VCorpus(VectorSource(dataset_original$Review))
as.character(corpus[1])
corpus = tm_map(corpus,content_transformer(tolower)) #This is used to convert to lower case
as.character(corpus[1])
as.character(corpus[841]) #We should remove numbers any in the dataset
corpus = tm_map(corpus,removeNumbers)
as.character(corpus[841])
corpus = tm_map(corpus,removePunctuation)
corpus = tm_map(corpus,removeWords,stopwords()) #Removes non relevant words
as.character(corpus[1])
corpus =tm_map(corpus, stemDocument)
corpus = tm_map(corpus,stripWhitespace)

#Create Bag of words

dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm,0.999)
dtm
dataset = as.data.frame(as.matrix(dtm))

#Artificial Neural networks
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]
dataset$Geography = as.numeric(factor(dataset$Geography, levels =c('France','Spain','Germany',
                                                                   labels=c(1,2,3))))
dataset$Gender = as.numeric(factor(dataset$Gender, levels =c('Female','Male'),
                                                                   labels=c(1,2)))

dataset

library(caTools)
#splitting the training and test set
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set= subset(dataset, split == TRUE)
test_set = subset(dataset, split==FALSE)

#Feature scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

#Fitting ANN to the training set
install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y ='Exited',training_frame = as.h2o(training_set),
                              activation ='Rectifier',hidden =c(6,6), epochs=100,
                              train_samples_per_iteration = -2)
#predicting the ANN output
prob_pred = h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
prob_pred
y_pred =prob_pred>0.5
y_pred= as.vector(y_pred)
y_pred
#Confusion matrix
cm = table(test_set[,11], y_pred)
cm
h2o.shutdown()

#Principal component analysis

dataset = read.csv('Wine.csv')
library(caTools)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset,split==TRUE)
test_set = subset(dataset,split==FALSE)

training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

library(caret)
library(e1071)
pca = preProcess(x = training_set[-14], method = 'pca',pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c(2,3,1)]
training_set

test_set = predict(pca, test_set)
test_set = test_set[c(2,3,1)]
test_set
classifier = svm(formula = Customer_Segment ~ .,data=training_set,
                 type = 'C-classification',kernel='linear')

y_pred = predict(classifier, new_data=test_set[-3])
y_pred

cm = table(test_set[, 3], y_pred)

#Linear Discriminant analysis

dataset = read.csv('Wine.csv')
library(caTools)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset,split==TRUE)
test_set = subset(dataset,split==FALSE)

training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

library(MASS)
lda = lda(formula= Customer_Segment~.,data = training_set)

training_set = predict(lda,training_set)
training_set = training_set[c(2,3,1)]
test_set = predict(lda,test_set)
test_set
classifier = svm(formula = Customer_Segment ~ .,data=training_set,
                 type = 'C-classification',kernel='linear')

y_pred = predict(classifier, new_data=test_set[-3])
y_pred

cm = table(test_set[, 3], y_pred)

install.packages('kernlab')
library('kernlab')

#Cross fold validation

dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]
dataset$Purchased =factor(dataset$Purchased,levels=c(0,1))

library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set= subset(dataset,split=TRUE)
test_set=subset(dataset,split=FALSE)
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2] = scale(test_set[,1:2])

library(caret)
library(e1071)
folds = createFolds(training_set$Purchased, k=10)
cv = lapply(folds, function(x){
  training_fold = training_set[-x, ]
  test_fold = training_set[x,]
  classifier = svm(formula = Purchased ~ .,data=training_fold,
                   type = 'C-classification',kernel='radial')
  y_pred = predict(classifier,new_data=test_fold[-3])
  cm=table(test_fold[,3],y_pred)
  accuracy =(cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
  return(accuracy)
  })





