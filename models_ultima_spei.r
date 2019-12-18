library(xtable) #for table creation for latex
library(ggplot2)#for graphics
library(MASS)#for lda
library(scales)#for scientific notation
library(RColorBrewer) #for base r plot
library(class) #for base r plot
library(plyr)#for obtaining means by factor
library(e1071)#for svm
library(nnet)#for multinomial regression

rm(list=ls())

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
rownames(model_rslts)<-c("CNN", "lda", "SVM", "LR")

##importing data for traditional image histograms
bird   <- read.table("bird.txt", sep=",", header=TRUE)
bone   <- read.table("bone.txt", sep=",", header=TRUE)
brick   <- read.table("brick.txt", sep=",", header=TRUE)
cam   <- read.table("cam.txt", sep=",", header=TRUE)
cup   <- read.table("cup.txt", sep=",", header=TRUE)

#cleaning data for ggplot2 and analysis
labs<-as.factor(sort(rep(1:5, 20)) )

mydata<-rbind(bird  ,
bone  ,
brick ,
cam   ,
cup)


#counts plot
temp<-as.data.frame(cbind(labs, mydata))
#labs2<-as.factor(c(rep("Edge", dim(edge)[1]),
#                  rep("Spiral", dim(spiral)[1]),
#                  rep("Ellipse", dim(elip)[1]) ))

#counts plot
temp<-as.data.frame(cbind(labs, mydata))
#labs2<-as.factor(c(rep("Edge", dim(edge)[1]),
#                  rep("Spiral", dim(spiral)[1]),
#                  rep("Ellipse", dim(elip)[1]) ))

scat<-ggplot(data=temp, aes(x = white, y = black, colour = as.factor(labs)))+
          geom_point(size=2)+
          #geom_ribbon(aes(ymin=temp$lower, ymax=temp$upper), linetype=2, alpha=0.1)+
	 	      ggtitle("EI for\nMPEG-7 Smallest")+
		      xlab("White Counts")+
					ylab("Black Counts")+
			 		labs(colour= "Legend")+
					scale_y_continuous(label=scientific_10)+
          scale_x_continuous(label=scientific_10)+
          mytheme.scat+
          #scale_color_discrete(breaks=c("Edge", "Spiral", "Ellipse"))+
          theme(legend.text=element_text(size=18),
                legend.title=element_text(size=24))

ggsave(filename="plots/Encircled_Image_Histograms_MPEG_7_Smallest.png", plot=scat,
       width=9, height=7)


n=4

#setup for validation plot

valid_results<-matrix(nrow=4, ncol=1, data=0)
colnames(valid_results)<-c("n=4")
rownames(valid_results)<-c("CNN", "lda", "SVM", "LR")

#setup for training plot
train_results<-matrix(nrow=4, ncol=1, data=0)
colnames(train_results)<-c("n=4")
rownames(train_results)<-c("CNN", "lda", "SVM", "LR")


##################################
## training sample size = 4
##################################

n=4

#cnn results for n=4
model_rslts[1,]<-c(1.00, 0.83)

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(9318)

#initialize objects to hold results
lda_train<-c()
lda_valid<-c()
svm_train<-c()
svm_valid<-c()
lr_train<-c()
lr_valid<-c()

#simuiltion size
sim=100

for (i in 1:sim) {

  train1<-sample(1:20,  n)
  train2<-sample(1:20, n)
  train3<-sample(1:20, n)
  train4<-sample(1:20, n)
  train5<-sample(1:20,  n)

  mytrain<-rbind(bird[train1,]  ,
  bone[train2,]  ,
  brick[train3,] ,
  cam[train4,]   ,
  cup[train5,])

  labs_train<-as.factor(sort(rep(1:5, n)) )
  myvalid<-rbind(bird[-train1,]  ,
  bone[-train2,]  ,
  brick[-train3,] ,
  cam[-train4,]   ,
  cup[-train5,])

  labs_valid<-as.factor(c(rep(1, 20-n), rep(2, 20-n),
                          rep(3, 20-n), rep(4, 20-n),
                          rep(5, 20-n)
                        ) )
  #######
  #lda
  #######
  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  #creating model
  lda.fit = lda(labs ~ white + black, data=train)
  #lda.fit #rank deficiency - ie unable to compute

  #predicting
  lda.pred=predict(lda.fit, train)
  lda.class = lda.pred$class

  #results
  #table(lda.class, labs_train)
  #overall classification rate for training
  lda_train[i]<- mean(lda.class==as.factor(as.numeric(labs_train)))

  ####
  #now predict on validation
  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #predicting
  lda.pred=predict(lda.fit, valid)
  lda.class = lda.pred$class

  #results
  #table(lda.class, labs_valid)
  #overall classification rate for training
  lda_valid[i]<-mean(lda.class==as.factor(as.numeric(labs_valid)))

  #######
  #SVM
  #######

  train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
  colnames(train)[1]<-"labs"

  valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
  colnames(valid)[1]<-"labs"

  #creating model
  svmfit=svm(labs ~ white + black, data=train, kernel="polynomial",
             cost=1000, coef0= 100, degree=2, gamma=1/150, #76 @ 1/100
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

    ################################
    #Multinomial Logistic Regression
    ################################

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    lr.fit=multinom(labs ~ white + black, data=train)

    ypred=predict(lr.fit ,train)
    #table(predict=ypred, truth=train$labs)
    lr_train[i]<-mean(ypred==as.factor(as.numeric(labs_train)))

    #now on valid
    ypred_valid=predict(lr.fit ,valid)
    #table(predict=ypred_valid, truth=valid$labs)
    lr_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}

#################
## Model Results
#################

#lda
model_rslts[2,1]<-mean(lda_train)
model_rslts[2,2]<-mean(lda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(lr_train)
model_rslts[4,2]<-mean(lr_valid)

sd(lda_train)
sd(lda_valid)
sd(svm_valid)
sd(svm_train)
sd(lr_train)
sd(lr_valid)


#display results
model_rslts

xtable(model_rslts, digits=2)

train_results[,1]<-model_rslts[,1]
valid_results[,1]<-model_rslts[,2]

#counts plot
sps<-mydata[,1]/rowSums(mydata)
aggregate(sps~labs, FUN=mean)
xtable(aggregate(sps~labs, FUN=sd))




#
