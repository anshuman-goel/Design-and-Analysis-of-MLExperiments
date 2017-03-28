# ========================================
# Multiple Hypothesis Testing
# Part 1: K-fold Cross-Validation Paired t-Test
# Part 2: Analysis of Variance (ANOVA) Test
# Part 3: Wilcoxon Signed Rank test
# ========================================

#Ref:
#http://support.minitab.com/en-us/minitab/17/topic-library/basic-statistics-and-graphs/hypothesis-tests/tests-of-means/why-use-paired-t/

# Load the required R packages

# install.packages('C50')
# install.packages('kernlab')
# install.packages('perry')
# install.packages('stats')
# install.packages('e1071')
library(C50)
library(kernlab)
library(perry)
library(e1071)
library(stats)

# **********************************************
# Part 1: K-fold Cross-Validation Paired t-Test
# *****************************************

# Load the iris data set

iris<-read.csv('E:\\NCSU\\Semester 2\\Algorithms for Data Guided Business Intelligence\\Homeworks\\Design and Analysis of MLExperiments\\datasets\\Iris_data.txt',header=FALSE,sep = ',')

# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds

train_iris_index<-sample(1:nrow(iris),nrow(iris))
random_iris<-iris[train_iris_index,]
iris_cvfolds<-cvFolds(nrow(random_iris),K=10)

# Use the training set to train a C5.0 decision tree and Support Vector Machine


# Make predictions on the test set and calculate the error percentages made by both the trained models
error_c5.0<-c()
error_svm<-c()

for(i in 1:10)
{
  train<-random_iris[iris_cvfolds$subsets[iris_cvfolds$which != i], ]
  test<-random_iris[iris_cvfolds$subsets[iris_cvfolds$which == i], ]
  
  #Decision Tree
  tree.model<-C5.0(train[,1:4],train[,5])
  
  #SVM
  #svm.model<-gausspr(V5~.,data=train)
  svm.model<-ksvm(V5~.,data=train)
  
  #Predicting
  tree.test<-predict(tree.model,newdata=test[,1:4])
  svm.test<-predict(svm.model,newdata=test[,1:4])
  
  #Error Calculation
  a<-tree.test==test[,5]
  b<-svm.test==test[,5]
  error_c5.0<-c(error_c5.0,length(a[a==FALSE])*100/length(a))
  error_svm<-c(error_svm,length(b[b==FALSE])*100/length(b))
}

print('C5.0 Error %')
print(error_c5.0)
print('SVM Error %')
print(error_svm)

# Perform K-fold Cross-Validation Paired t-Test to compare the means of the two error percentages

t.test(error_c5.0,error_svm,paired=TRUE)

# *****************************************
# Part 2: Analysis of Variance (ANOVA) Test
# *****************************************

# Load the Breast Cancer data set 

breast<-read.csv('E:\\NCSU\\Semester 2\\Algorithms for Data Guided Business Intelligence\\Homeworks\\Design and Analysis of MLExperiments\\datasets\\Wisconsin_Breast_Cancer_data.txt',header=FALSE,sep = ',')
breast<-data.frame(breast[,3:32],breast[,2:2])
colnames(breast)[31]<-"V33"

# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds

random_breast<-breast[sample(1:nrow(breast),nrow(breast)),]
breast_cvfolds<-cvFolds(nrow(random_breast),K=10)

# Use the training set to train following classifier algorithms
# 	1. C5.0 decision tree (see ?C5.0 in C50 package)
# 	2. Support Vector Machine (see ?ksvm in kernlab package)
# 	3. Naive Bayes	(?naiveBayes in e1071 package) 
# 	4. Logistic Regression (?glm in stats package) 


# Make predictions on the test set and calculate the error percentages made by the trained models

error_c5.0<-c()
error_svm<-c()
error_nb<-c()
error_glm<-c()

for(i in 1:10)
{
  train<-random_breast[breast_cvfolds$subsets[breast_cvfolds$which != i], ]
  test<-random_breast[breast_cvfolds$subsets[breast_cvfolds$which == i], ]
  
  #C5.0
  tree.model<-C5.0(train[,1:30],train[,31])
  
  #SVM
  svm.model<-ksvm(V33~.,data = train)
  
  #NB
  nb.model<-naiveBayes(train[,1:30],train[,31])
  
  #LR
  glm.model<-glm(V33~.,family=binomial(),data = train)
  
  #Predicting
  tree.test<-predict(tree.model,newdata=test[,1:30])
  svm.test<-predict(svm.model,newdata=test[,1:30])
  nb.test<-predict(nb.model,newdata=test[,1:30])
  glm.test<-predict(glm.model,newdata=test[,1:30], type = "response")
  
  #Error Calculation
  a<-tree.test==test[,31]
  b<-svm.test==test[,31]
  c<-nb.test==test[,31]
  d<-glm.test<=0.5
  error_c5.0<-c(error_c5.0,length(a[a==FALSE])*100/length(a))
  error_svm<-c(error_svm,length(b[b==FALSE])*100/length(b))
  error_nb<-c(error_nb,length(c[c==FALSE])*100/length(c))
  error_glm<-c(error_glm,length(d[d==FALSE])*100/length(d))
}

print('C5.0 Error %')
print(error_c5.0)
print('SVM Error %')
print(error_svm)
print('NB Error %')
print(error_nb)
print('GLM Error %')
print(error_glm)

# Compare the performance of the different classifiers using ANOVA test (see ?aov)

#Ref:
#http://www.sthda.com/english/wiki/one-way-anova-test-in-r
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3916511/
#https://www.r-bloggers.com/analysis-of-variance-anova-for-multiple-comparisons/

dat<-c(error_c5.0,error_svm,error_nb,error_glm)
group<-factor(rep(c("c5.0","svm","nb","glm"), each = 10))
aov.test<-aov(dat ~ group)
summary(aov.test)
TukeyHSD(aov.test)

# *****************************************
# Part 3: Wilcoxon Signed Rank test
# *****************************************

# Load the following data sets,
# 1. Iris 

iris<-read.csv('E:\\NCSU\\Semester 2\\Algorithms for Data Guided Business Intelligence\\Homeworks\\Design and Analysis of MLExperiments\\datasets\\Iris_data.txt',header=FALSE,sep = ',')

# 2. Ecoli 

ecoli<-read.csv('E:\\NCSU\\Semester 2\\Algorithms for Data Guided Business Intelligence\\Homeworks\\Design and Analysis of MLExperiments\\datasets\\Ecoli_data.csv',header=FALSE,sep = ',')

# 3. Wisconsin Breast Cancer

breast<-read.csv('E:\\NCSU\\Semester 2\\Algorithms for Data Guided Business Intelligence\\Homeworks\\Design and Analysis of MLExperiments\\datasets\\Wisconsin_Breast_Cancer_data.txt',header=FALSE,sep = ',')
breast<-data.frame(breast[,3:32],breast[,2:2])
colnames(breast)[31]<-"V33"

# 4. Glass

glass<-read.csv('E:\\NCSU\\Semester 2\\Algorithms for Data Guided Business Intelligence\\Homeworks\\Design and Analysis of MLExperiments\\datasets\\Glass_data.txt',header=FALSE,sep = ',')
glass<-glass[,2:ncol(glass)]

# 5. Yeast

yeast<-read.csv('E:\\NCSU\\Semester 2\\Algorithms for Data Guided Business Intelligence\\Homeworks\\Design and Analysis of MLExperiments\\datasets\\Yeast_data.csv',header=FALSE,sep = ',')

# Randomize the data and perform 10-fold Cross-Validation
# See ?sample and ?cvFolds

#Iris

random_iris<-iris[sample(1:nrow(iris),nrow(iris)),]
iris_cvfolds<-cvFolds(nrow(random_iris),K=10)

#Ecoli

random_ecoli<-ecoli[sample(1:nrow(ecoli),nrow(ecoli)),]
ecoli_cvfolds<-cvFolds(nrow(random_ecoli),K=10)

#Breast Cancer

random_breast<-breast[sample(1:nrow(breast),nrow(breast)),]
breast_cvfolds<-cvFolds(nrow(random_breast),K=10)

#Glass

random_glass<-glass[sample(1:nrow(glass),nrow(glass)),]
glass_cvfolds<-cvFolds(nrow(random_glass),K=10)

#Yeast

random_yeast<-yeast[sample(1:nrow(yeast),nrow(yeast)),]
yeast_cvfolds<-cvFolds(nrow(random_yeast),K=10)

# Use the training set to train following classifier algorithms
# 	1. C5.0 decision tree (see ?C5.0 in C50 package)
# 	2. Support Vector Machine (see ?ksvm in kernlab package)

# Make predictions on the test set and calculate the error percentages made by the trained models

error_c5.0_iris<-c()
error_svm_iris<-c()

error_c5.0_ecoli<-c()
error_svm_ecoli<-c()

error_c5.0_breast<-c()
error_svm_breast<-c()

error_c5.0_glass<-c()
error_svm_glass<-c()

error_c5.0_yeast<-c()
error_svm_yeast<-c()

for(i in 1:10)
{
  train_iris<-random_iris[iris_cvfolds$subsets[iris_cvfolds$which != i], ]
  test_iris<-random_iris[iris_cvfolds$subsets[iris_cvfolds$which == i], ]

  train_ecoli<-random_ecoli[ecoli_cvfolds$subsets[ecoli_cvfolds$which != i], ]
  test_ecoli<-random_ecoli[ecoli_cvfolds$subsets[ecoli_cvfolds$which == i], ]
  
  train_breast<-random_breast[breast_cvfolds$subsets[breast_cvfolds$which != i], ]
  test_breast<-random_breast[breast_cvfolds$subsets[breast_cvfolds$which == i], ]
  
  train_glass<-random_glass[glass_cvfolds$subsets[glass_cvfolds$which != i], ]
  test_glass<-random_glass[glass_cvfolds$subsets[glass_cvfolds$which == i], ]
  
  train_yeast<-random_yeast[yeast_cvfolds$subsets[yeast_cvfolds$which != i], ]
  test_yeast<-random_yeast[yeast_cvfolds$subsets[yeast_cvfolds$which == i], ]
  
  #Iris
  #Decision Tree
  tree.model_iris<-C5.0(train_iris[,1:4],train_iris[,5])
  
  #SVM
  svm.model_iris<-ksvm(V5~.,data=train_iris)
  
  #Ecoli
  #Decision Tree
  tree.model_ecoli<-C5.0(train_ecoli[,1:8],train_ecoli[,9])
  
  #SVM
  svm.model_ecoli<-ksvm(V9~.,data=train_ecoli)
  
  #Breast
  #Decision Tree
  tree.model_breast<-C5.0(train_breast[,1:(ncol(train_breast)-1)],train_breast[,ncol(train_breast)])
  
  #SVM
  svm.model_breast<-ksvm(V33~.,data=train_breast)
  
  #Glass
  #Decision Tree
  tree.model_glass<-C5.0(train_glass[,1:9],as.factor(train_glass[,10]))
  
  #SVM
  svm.model_glass<-ksvm(as.factor(train_glass[,10])~.,data=train_glass)
  
  #Yeast
  #Decision Tree
  tree.model_yeast<-C5.0(train_yeast[,1:9],train_yeast[,10])
  
  #SVM
  svm.model_yeast<-ksvm(V10~.,data=train_yeast)
  
  #Predicting
  #Iris
  tree.test_iris<-predict(tree.model_iris,newdata=test_iris[,1:4])
  svm.test_iris<-predict(svm.model_iris,newdata=test_iris[,1:4])
  
  #Ecoli
  tree.test_ecoli<-predict(tree.model_ecoli,newdata=test_ecoli[,1:8])
  svm.test_ecoli<-predict(svm.model_ecoli,newdata=test_ecoli[,1:8])
  
  #Breast
  tree.test_breast<-predict(tree.model_breast,newdata=test_breast[,1:(ncol(test_breast)-1)])
  svm.test_breast<-predict(svm.model_breast,newdata=test_breast[,1:(ncol(test_breast)-1)])
  
  #Glass
  tree.test_glass<-predict(tree.model_glass,newdata=test_glass[,1:9])
  svm.test_glass<-predict(svm.model_glass,newdata=test_glass)
  
  #Yeast
  tree.test_yeast<-predict(tree.model_yeast,newdata=test_yeast[,1:9])
  svm.test_yeast<-predict(svm.model_yeast,newdata=test_yeast[,1:9])
  
  #Error Calculation
  #Iris
  a<-tree.test_iris==test_iris[,5]
  b<-svm.test_iris==test_iris[,5]
  error_c5.0_iris<-c(error_c5.0_iris,length(a[a==FALSE])*100/length(a))
  error_svm_iris<-c(error_svm_iris,length(b[b==FALSE])*100/length(b))
  
  #Ecoli
  a<-tree.test_ecoli==test_ecoli[,9]
  b<-svm.test_ecoli==test_ecoli[,9]
  error_c5.0_ecoli<-c(error_c5.0_ecoli,length(a[a==FALSE])*100/length(a))
  error_svm_ecoli<-c(error_svm_ecoli,length(b[b==FALSE])*100/length(b))
  
  #Breast
  a<-tree.test_breast==test_breast[,ncol(test_breast)]
  b<-svm.test_breast==test_breast[,ncol(test_breast)]
  error_c5.0_breast<-c(error_c5.0_breast,length(a[a==FALSE])*100/length(a))
  error_svm_breast<-c(error_svm_breast,length(b[b==FALSE])*100/length(b))
  
  #Glass
  a<-tree.test_glass==test_glass[,10]
  b<-svm.test_glass==test_glass[,10]
  error_c5.0_glass<-c(error_c5.0_glass,length(a[a==FALSE])*100/length(a))
  error_svm_glass<-c(error_svm_glass,length(b[b==FALSE])*100/length(b))
  
  #Yeast
  a<-tree.test_yeast==test_yeast[,10]
  b<-svm.test_yeast==test_yeast[,10]
  error_c5.0_yeast<-c(error_c5.0_yeast,length(a[a==FALSE])*100/length(a))
  error_svm_yeast<-c(error_svm_yeast,length(b[b==FALSE])*100/length(b))
}

print('Iris C5.0 Error %')
print(error_c5.0_iris)
print('Iris SVM Error %')
print(error_svm_iris)

print('Ecoli C5.0 Error %')
print(error_c5.0_ecoli)
print('Ecoli SVM Error %')
print(error_svm_ecoli)

print('Breast C5.0 Error %')
print(error_c5.0_breast)
print('Breast SVM Error %')
print(error_svm_breast)

print('Glass C5.0 Error %')
print(error_c5.0_glass)
print('Glass SVM Error %')
print(error_svm_glass)

print('Yeast C5.0 Error %')
print(error_c5.0_yeast)
print('Yeast SVM Error %')
print(error_svm_yeast)

# Compare the performance of the different classifiers using Wilcoxon Signed Rank test (see ?wilcox.test)
#Ref:
#http://www.sthda.com/english/wiki/unpaired-two-samples-wilcoxon-test-in-r

mean_error_c5.0<-c(mean(error_c5.0_iris),mean(error_c5.0_ecoli),
                   mean(error_c5.0_breast),mean(error_c5.0_glass),
                   mean(error_c5.0_yeast))
mean_error_svm<-c(mean(error_svm_iris),mean(error_svm_ecoli),
                   mean(error_svm_breast),mean(error_svm_glass),
                   mean(error_svm_yeast))
wilcox.test(mean_error_c5.0,mean_error_svm,paired = TRUE)