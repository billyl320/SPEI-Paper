library(xtable) #for table creation for latex
library(ggplot2)#for graphics
library(MASS)#for qda
library(scales)#for scientific notation
library(RColorBrewer) #for base r plot
library(class) #for base r plot
library(plyr)#for obtaining means by factor
library(e1071)#for svm
library(tree)#for tree based methods

#defining proper scientific notation

scientific_10 <- function(x) {
  parse(text=gsub("e", " %*% 10^", scales::scientific_format()(x)))
}

#custom theme
mytheme.scat<-theme(

	plot.title = element_text(size=60, face="bold", hjust = 0.5),
	axis.text.x  = element_text(size=20, face="bold"),
	axis.text.y=element_text(size=20, face="bold"),
	axis.title.x=element_text(size=28, face='bold'),
	axis.title.y=element_text(size=28, face='bold'),
	strip.background=element_rect(fill="gray80"),
	panel.background=element_rect(fill="gray80"),
	axis.ticks= element_blank(),
	axis.text=element_text(colour="black"),
  strip.text = element_text(size=25)

	)


#matrix to hold results
model_rslts<-matrix(nrow=4, ncol=2, data=0)
colnames(model_rslts)<-c("Train", "Validation")
rownames(model_rslts)<-c("CNN", "QDA", "SVM", "Tree")

##importing data for traditional image histograms
edge   <- read.table("edge.txt", sep=",", header=TRUE)
spiral <- read.table("spiral.txt", sep=",", header=TRUE)
elip   <- read.table("elip.txt", sep=",", header=TRUE)

#cleaning data for ggplot2 and analysis
labs<-as.factor(c(rep(1, dim(edge)[1]), rep(2, dim(spiral)[1]), rep(3, dim(elip)[1])) )

mydata<-rbind(edge, spiral, elip)


#counts plot
temp<-as.data.frame(cbind(labs, mydata))
labs2<-as.factor(c(rep("Edge", dim(edge)[1]), rep("Spiral", dim(spiral)[1]), rep("Ellipse", dim(elip)[1]) ))

scat<-ggplot(data=temp, aes(x = white, y = black, colour = as.factor(labs2)))+
          geom_point(size=2)+
          #geom_ribbon(aes(ymin=temp$lower, ymax=temp$upper), linetype=2, alpha=0.1)+
	 	      ggtitle("EI for\nGalaxy Shapes")+
		      xlab("White Counts")+
					ylab("Black Counts")+
			 		labs(colour= "Legend")+
					scale_y_continuous(label=scientific_10)+
          scale_x_continuous(label=scientific_10)+
          mytheme.scat+
          scale_color_discrete(breaks=c("Edge", "Spiral", "Ellipse"))+
          theme(legend.text=element_text(size=18),
                legend.title=element_text(size=24))

ggsave(filename="plots/Encircled_Image_Histograms_galaxy.png", plot=scat,
       width=9, height=7)


#

#setup for validation plot

valid_results<-matrix(nrow=4, ncol=6, data=0)
colnames(valid_results)<-c("n=3", "n=4", "n=5", "n=7", "n=10", "n=20")
rownames(valid_results)<-c("CNN", "QDA", "SVM", "Tree")

#setup for training plot
train_results<-matrix(nrow=4, ncol=6, data=0)
colnames(train_results)<-c("n=3", "n=4", "n=5", "n=7", "n=10", "n=20")
rownames(train_results)<-c("CNN", "QDA", "SVM", "Tree")

##################################
## training sample size = 3
##################################

n=3

#cnn results for n=3
model_rslts[1,]<-c(1.00, 0.55)

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(76526)

#initialize objects to hold results
qda_train<-c()
qda_valid<-c()
svm_train<-c()
svm_valid<-c()
tree_train<-c()
tree_valid<-c()

#simuiltion size
sim=100

for (i in 1:sim) {

  train1<-sample(1:75,  n)
  train2<-sample(1:223, n)
  train3<-sample(1:225, n)

  mytrain<-rbind(edge[train1,], spiral[train2,],
                 elip[train3,])
  labs_train<-as.factor(c(rep(1, n), rep(2, n),
                          rep(3, n) ) )
  myvalid<-rbind(edge[-train1,], spiral[-train2,],
                 elip[-train3,])
  labs_valid<-as.factor(c(rep(1, 75-n), rep(2, 225-n),
                          rep(3, 223-n) ) )

  #######
  #QDA
  #######
  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  #creating model
  qda.fit = qda(labs ~ white + black, data=train)
  #qda.fit #rank deficiency - ie unable to compute

  #predicting
  qda.pred=predict(qda.fit, train)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_train)
  #overall classification rate for training
  qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

  ####
  #now predict on validation
  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #predicting
  qda.pred=predict(qda.fit, valid)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_valid)
  #overall classification rate for training
  qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

  #######
  #SVM
  #######

  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #creating model
  svmfit=svm(labs ~ white + black, data=train, kernel="linear",
             cost=2,#, coef0= 1, degree=2,
             scale=FALSE)

  #plot(svmfit , train)

  #summary(svmfit)

  ypred=predict(svmfit ,train)
  #table(predict=ypred, truth=train$labs)
  svm_train[i]<-mean(ypred==as.factor(as.numeric(labs_train)))

  #now on valid
  ypred_valid=predict(svmfit ,valid)
  #table(predict=ypred_valid, truth=valid$labs)
  svm_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

  ######
  # Tree
  #######

  #training tree mdoel
  treefit =tree(labs ~ white + black, data=train )
  #summary(treefit)

  ypred_train=predict(treefit ,train, type='class')
  #table(predict=ypred_train, truth=as.factor(train$labs))
  tree_train[i]<-mean(ypred_train==as.factor(as.numeric(labs_train)))

  #plot(treefit )
  #text(treefit ,pretty =0)

  ypred_valid=predict(treefit ,valid, type='class')
  #table(predict=ypred_valid, truth=valid$labs)
  tree_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}

#################
## Model Results
#################

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)


#display results
model_rslts

xtable(model_rslts, digits=2)

valid_results[,1]<-model_rslts[,2]
train_results[,1]<-model_rslts[,1]

##################################
## training sample size = 4
##################################

n=4

#cnn results for n=4
model_rslts[1,]<-c(1.00, 0.56)

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(872596)

#initialize objects to hold results
qda_train<-c()
qda_valid<-c()
svm_train<-c()
svm_valid<-c()
tree_train<-c()
tree_valid<-c()

#simuiltion size
sim=100

for (i in 1:sim) {

  train1<-sample(1:75,  n)
  train2<-sample(1:223, n)
  train3<-sample(1:225, n)

  mytrain<-rbind(edge[train1,], spiral[train2,],
                 elip[train3,])
  labs_train<-as.factor(c(rep(1, n), rep(2, n),
                          rep(3, n) ) )
  myvalid<-rbind(edge[-train1,], spiral[-train2,],
                 elip[-train3,])
  labs_valid<-as.factor(c(rep(1, 75-n), rep(2, 225-n),
                          rep(3, 223-n) ) )

  #######
  #QDA
  #######
  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  #creating model
  qda.fit = qda(labs ~ white + black, data=train)
  #qda.fit #rank deficiency - ie unable to compute

  #predicting
  qda.pred=predict(qda.fit, train)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_train)
  #overall classification rate for training
  qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

  ####
  #now predict on validation
  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #predicting
  qda.pred=predict(qda.fit, valid)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_valid)
  #overall classification rate for training
  qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

  #######
  #SVM
  #######

  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #creating model
  svmfit=svm(labs ~ white + black, data=train, kernel="linear",
             cost=2,#, coef0= 1, degree=2,
             scale=FALSE)

  #plot(svmfit , train)

  #summary(svmfit)

  ypred=predict(svmfit ,train)
  #table(predict=ypred, truth=train$labs)
  svm_train[i]<-mean(ypred==as.factor(as.numeric(labs_train)))

  #now on valid
  ypred_valid=predict(svmfit ,valid)
  #table(predict=ypred_valid, truth=valid$labs)
  svm_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

  ######
  # Tree
  #######

  #training tree mdoel
  treefit =tree(labs ~ white + black, data=train )
  #summary(treefit)

  ypred_train=predict(treefit ,train, type='class')
  #table(predict=ypred_train, truth=as.factor(train$labs))
  tree_train[i]<-mean(ypred_train==as.factor(as.numeric(labs_train)))

  #plot(treefit )
  #text(treefit ,pretty =0)

  ypred_valid=predict(treefit ,valid, type='class')
  #table(predict=ypred_valid, truth=valid$labs)
  tree_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}

#################
## Model Results
#################

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)


#display results
model_rslts

xtable(model_rslts, digits=2)

train_results[,2]<-model_rslts[,1]
valid_results[,2]<-model_rslts[,2]

##################################
## training sample size = 5
##################################

n=5

#cnn results for n=5
model_rslts[1,]<-c(1.00, 0.56)

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(50976)

#initialize objects to hold results
qda_train<-c()
qda_valid<-c()
svm_train<-c()
svm_valid<-c()
tree_train<-c()
tree_valid<-c()

#simuiltion size
sim=100

for (i in 1:sim) {

  train1<-sample(1:75,  n)
  train2<-sample(1:223, n)
  train3<-sample(1:225, n)

  mytrain<-rbind(edge[train1,], spiral[train2,],
                 elip[train3,])
  labs_train<-as.factor(c(rep(1, n), rep(2, n),
                          rep(3, n) ) )
  myvalid<-rbind(edge[-train1,], spiral[-train2,],
                 elip[-train3,])
  labs_valid<-as.factor(c(rep(1, 75-n), rep(2, 225-n),
                          rep(3, 223-n) ) )

  #######
  #QDA
  #######
  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  #creating model
  qda.fit = qda(labs ~ white + black, data=train)
  #qda.fit #rank deficiency - ie unable to compute

  #predicting
  qda.pred=predict(qda.fit, train)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_train)
  #overall classification rate for training
  qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

  ####
  #now predict on validation
  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #predicting
  qda.pred=predict(qda.fit, valid)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_valid)
  #overall classification rate for training
  qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

  #######
  #SVM
  #######

  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #creating model
  svmfit=svm(labs ~ white + black, data=train, kernel="linear",
             cost=2,#, coef0= 1, degree=2,
             scale=FALSE)

  #plot(svmfit , train)

  #summary(svmfit)

  ypred=predict(svmfit ,train)
  #table(predict=ypred, truth=train$labs)
  svm_train[i]<-mean(ypred==as.factor(as.numeric(labs_train)))

  #now on valid
  ypred_valid=predict(svmfit ,valid)
  #table(predict=ypred_valid, truth=valid$labs)
  svm_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

  ######
  # Tree
  #######

  #training tree mdoel
  treefit =tree(labs ~ white + black, data=train )
  #summary(treefit)

  ypred_train=predict(treefit ,train, type='class')
  #table(predict=ypred_train, truth=as.factor(train$labs))
  tree_train[i]<-mean(ypred_train==as.factor(as.numeric(labs_train)))

  #plot(treefit )
  #text(treefit ,pretty =0)

  ypred_valid=predict(treefit ,valid, type='class')
  #table(predict=ypred_valid, truth=valid$labs)
  tree_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}

#################
## Model Results
#################

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)


#display results
model_rslts

xtable(model_rslts, digits=2)

train_results[,3]<-model_rslts[,1]
valid_results[,3]<-model_rslts[,2]

##################################
## training sample size = 7
##################################

n=7

#cnn results for n=7
model_rslts[1,]<-c(1.00, 0.58)

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(35522)

#initialize objects to hold results
qda_train<-c()
qda_valid<-c()
svm_train<-c()
svm_valid<-c()
tree_train<-c()
tree_valid<-c()

#simuiltion size
sim=100

for (i in 1:sim) {

  train1<-sample(1:75,  n)
  train2<-sample(1:223, n)
  train3<-sample(1:225, n)

  mytrain<-rbind(edge[train1,], spiral[train2,],
                 elip[train3,])
  labs_train<-as.factor(c(rep(1, n), rep(2, n),
                          rep(3, n) ) )
  myvalid<-rbind(edge[-train1,], spiral[-train2,],
                 elip[-train3,])
  labs_valid<-as.factor(c(rep(1, 75-n), rep(2, 225-n),
                          rep(3, 223-n) ) )

  #######
  #QDA
  #######
  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  #creating model
  qda.fit = qda(labs ~ white + black, data=train)
  #qda.fit #rank deficiency - ie unable to compute

  #predicting
  qda.pred=predict(qda.fit, train)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_train)
  #overall classification rate for training
  qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

  ####
  #now predict on validation
  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #predicting
  qda.pred=predict(qda.fit, valid)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_valid)
  #overall classification rate for training
  qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

  #######
  #SVM
  #######

  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #creating model
  svmfit=svm(labs ~ white + black, data=train, kernel="linear",
             cost=2,#, coef0= 1, degree=2,
             scale=FALSE)

  #plot(svmfit , train)

  #summary(svmfit)

  ypred=predict(svmfit ,train)
  #table(predict=ypred, truth=train$labs)
  svm_train[i]<-mean(ypred==as.factor(as.numeric(labs_train)))

  #now on valid
  ypred_valid=predict(svmfit ,valid)
  #table(predict=ypred_valid, truth=valid$labs)
  svm_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

  ######
  # Tree
  #######

  #training tree mdoel
  treefit =tree(labs ~ white + black, data=train )
  #summary(treefit)

  ypred_train=predict(treefit ,train, type='class')
  #table(predict=ypred_train, truth=as.factor(train$labs))
  tree_train[i]<-mean(ypred_train==as.factor(as.numeric(labs_train)))

  #plot(treefit )
  #text(treefit ,pretty =0)

  ypred_valid=predict(treefit ,valid, type='class')
  #table(predict=ypred_valid, truth=valid$labs)
  tree_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}

#################
## Model Results
#################

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)


#display results
model_rslts

xtable(model_rslts, digits=2)

train_results[,4]<-model_rslts[,1]
valid_results[,4]<-model_rslts[,2]

##################################
## training sample size = 10
##################################

n=10

#cnn results for n=10
model_rslts[1,]<-c(1.00, 0.60)

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(1275148)

#initialize objects to hold results
qda_train<-c()
qda_valid<-c()
svm_train<-c()
svm_valid<-c()
tree_train<-c()
tree_valid<-c()

#simuiltion size
sim=100

for (i in 1:sim) {

  train1<-sample(1:75,  n)
  train2<-sample(1:223, n)
  train3<-sample(1:225, n)

  mytrain<-rbind(edge[train1,], spiral[train2,],
                 elip[train3,])
  labs_train<-as.factor(c(rep(1, n), rep(2, n),
                          rep(3, n) ) )
  myvalid<-rbind(edge[-train1,], spiral[-train2,],
                 elip[-train3,])
  labs_valid<-as.factor(c(rep(1, 75-n), rep(2, 225-n),
                          rep(3, 223-n) ) )

  #######
  #QDA
  #######
  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  #creating model
  qda.fit = qda(labs ~ white + black, data=train)
  #qda.fit #rank deficiency - ie unable to compute

  #predicting
  qda.pred=predict(qda.fit, train)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_train)
  #overall classification rate for training
  qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

  ####
  #now predict on validation
  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #predicting
  qda.pred=predict(qda.fit, valid)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_valid)
  #overall classification rate for training
  qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

  #######
  #SVM
  #######

  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #creating model
  svmfit=svm(labs ~ white + black, data=train, kernel="linear",
             cost=2,#, coef0= 1, degree=2,
             scale=FALSE)

  #plot(svmfit , train)

  #summary(svmfit)

  ypred=predict(svmfit ,train)
  #table(predict=ypred, truth=train$labs)
  svm_train[i]<-mean(ypred==as.factor(as.numeric(labs_train)))

  #now on valid
  ypred_valid=predict(svmfit ,valid)
  #table(predict=ypred_valid, truth=valid$labs)
  svm_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

  ######
  # Tree
  #######

  #training tree mdoel
  treefit =tree(labs ~ white + black, data=train )
  #summary(treefit)

  ypred_train=predict(treefit ,train, type='class')
  #table(predict=ypred_train, truth=as.factor(train$labs))
  tree_train[i]<-mean(ypred_train==as.factor(as.numeric(labs_train)))

  #plot(treefit )
  #text(treefit ,pretty =0)

  ypred_valid=predict(treefit ,valid, type='class')
  #table(predict=ypred_valid, truth=valid$labs)
  tree_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}

#################
## Model Results
#################

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)


#display results
model_rslts

xtable(model_rslts, digits=2)

train_results[,5]<-model_rslts[,1]
valid_results[,5]<-model_rslts[,2]

##################################
## training sample size = 20
##################################

n=20

#cnn results for n=29
model_rslts[1,]<-c(1.00, 0.64)

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(5924544)

#initialize objects to hold results
qda_train<-c()
qda_valid<-c()
svm_train<-c()
svm_valid<-c()
tree_train<-c()
tree_valid<-c()

#simuiltion size
sim=100

for (i in 1:sim) {

  train1<-sample(1:75,  n)
  train2<-sample(1:223, n)
  train3<-sample(1:225, n)

  mytrain<-rbind(edge[train1,], spiral[train2,],
                 elip[train3,])
  labs_train<-as.factor(c(rep(1, n), rep(2, n),
                          rep(3, n) ) )
  myvalid<-rbind(edge[-train1,], spiral[-train2,],
                 elip[-train3,])
  labs_valid<-as.factor(c(rep(1, 75-n), rep(2, 225-n),
                          rep(3, 223-n) ) )

  #######
  #QDA
  #######
  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  #creating model
  qda.fit = qda(labs ~ white + black, data=train)
  #qda.fit #rank deficiency - ie unable to compute

  #predicting
  qda.pred=predict(qda.fit, train)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_train)
  #overall classification rate for training
  qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

  ####
  #now predict on validation
  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #predicting
  qda.pred=predict(qda.fit, valid)
  qda.class = qda.pred$class

  #results
  #table(qda.class, labs_valid)
  #overall classification rate for training
  qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

  #######
  #SVM
  #######

  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #creating model
  svmfit=svm(labs ~ white + black, data=train, kernel="linear",
             cost=2,#, coef0= 1, degree=2,
             scale=FALSE)

  #plot(svmfit , train)

  #summary(svmfit)

  ypred=predict(svmfit ,train)
  #table(predict=ypred, truth=train$labs)
  svm_train[i]<-mean(ypred==as.factor(as.numeric(labs_train)))

  #now on valid
  ypred_valid=predict(svmfit ,valid)
  #table(predict=ypred_valid, truth=valid$labs)
  svm_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

  ######
  # Tree
  #######

  #training tree mdoel
  treefit =tree(labs ~ white + black, data=train )
  #summary(treefit)

  ypred_train=predict(treefit ,train, type='class')
  #table(predict=ypred_train, truth=as.factor(train$labs))
  tree_train[i]<-mean(ypred_train==as.factor(as.numeric(labs_train)))

  #plot(treefit )
  #text(treefit ,pretty =0)

  ypred_valid=predict(treefit ,valid, type='class')
  #table(predict=ypred_valid, truth=valid$labs)
  tree_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}

#################
## Model Results
#################

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)


#display results
model_rslts

xtable(model_rslts, digits=2)

train_results[,6]<-model_rslts[,1]
valid_results[,6]<-model_rslts[,2]

train_results

valid_results

xtable(valid_results)

xtable(train_results)

ultima<-as.data.frame(rbind(train_results, valid_results))

fcts<-as.factor(c(rep(1, 4), rep(2, 4)))

ultima<-cbind(ultima, fcts)

ultima

xtable(ultima)


#final results plot

models<-( rep(rownames(ultima)[1:4], 6*4 ) )
set<-( rep(c(rep("Training", 4), rep("Validation", 4)), 6) )
acc<-c(ultima[,1], ultima[,2], ultima[,3],
       ultima[,4], ultima[,5], ultima[,6])
samp<-c( rep(3.0, 8), rep(4.0, 8), rep(5.0, 8),
         rep(7.0, 8), rep(10.0, 8), rep(20.0, 8))
mydata<-as.data.frame(cbind(models, (acc), set, as.numeric(samp) ) )

colnames(mydata)[2]<-"Acc"
colnames(mydata)[4]<-"Samp"


ultima_plot<-ggplot(data=mydata,
            aes(x = as.numeric(as.character(mydata$Samp)),
                y = as.numeric(as.character(mydata$Acc)),
                colour = as.factor(mydata$models),
                shape= as.factor(mydata$set),
                linetype= as.factor(mydata$set),
                group=interaction(as.factor(mydata$models), as.factor(mydata$set))
                ) )+
          geom_point(size=4)+
          geom_line(size=2 )+
          #geom_ribbon(aes(ymin=temp$lower, ymax=temp$upper), linetype=2, alpha=0.1)+
	 	  ggtitle("Overall Results for\nGalaxy Shapes")+
		  xlab("Training Size")+
		  ylab("Overall Accuracy")+
		  labs(colour= "Model", shape="Data Set", linetype="Data Set")+
	      #scale_y_discrete(limits=c(0, 1.00))+
          #scale_x_discrete(breaks=c(3, 4, 5, 7, 10, 20))+
          mytheme.scat+
          scale_colour_manual(values = c("Red", "Blue", "Green", "khaki2"))+
          #scale_color_discrete(breaks=c("Training", "Validation"))+
          theme(legend.text=element_text(size=18),
                legend.title=element_text(size=24))

ultima_plot

ggsave(filename="plots/OverallAcc_galaxy.png", plot=ultima_plot,
       width=9, height=7)


##########################
# Empirical SP Estimation
##########################


labs<-as.factor(c(rep(1, dim(edge)[1]), rep(2, dim(spiral)[1]), rep(3, dim(elip)[1])) )
mydata<-rbind(edge, spiral, elip)


#counts plot
sps<-mydata[,1]/rowSums(mydata)
aggregate(sps~labs, FUN=mean)
xtable(aggregate(sps~labs, FUN=sd))




#
