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

#getting theoretical values
n <-c(3:8)

#matrix to hold results
model_rslts<-matrix(nrow=3, ncol=2, data=0)
colnames(model_rslts)<-c("Train", "Validation")
rownames(model_rslts)<-c("CNN", "SVM", "Tree")

model_rslts[1,]<-c(1.00, 0.80)

#importing data for encircled image histograms
tris <- read.table("tris.txt", sep=",", header=TRUE)
squs <- read.table("squs.txt", sep=",", header=TRUE)
pens <- read.table("pens.txt", sep=",", header=TRUE)
pens2<- read.table('non_reg_pens.txt', sep=",", header=TRUE)
hexs <- read.table("hexs.txt", sep=",", header=TRUE)
hexs2<- read.table('non_reg_hexs.txt', sep=",", header=TRUE)


#cleaning data for ggplot2 and analysis
labs<-as.factor(c(rep(1, dim(tris)[1]), rep(2, dim(squs)[1]),
                  rep(3, dim(pens)[1]), rep(3, dim(pens2)[1]),
                  rep(4, dim(hexs)[1]), rep(4, dim(hexs2)[1]) ) )

mydata<-rbind(tris, squs, pens, pens2, hexs, hexs2)

#counts plot
temp<-as.data.frame(cbind(labs, mydata))
labs2<-as.factor(c(rep("n=3", dim(tris)[1]), rep("n=4", dim(squs)[1]), rep("n=5", dim(pens)[1]),
                rep("n=5", dim(pens2)[1]), rep("n=6", dim(hexs)[1]),   rep("n=6", dim(hexs2)[1]) ))


scat<-ggplot(data=temp, aes(x = white, y = black, colour = as.factor(labs2)))+
          geom_point(size=2)+
	 	      ggtitle("Encircled\nImage Histogram")+
		      xlab("White Counts")+
					ylab("Black Counts")+
			 		labs(colour= "Legend")+
					scale_y_continuous(label=scientific_10)+
          scale_x_continuous(label=scientific_10)+
          mytheme.scat+
          scale_color_discrete(breaks=c("n=3","n=4","n=5", "n=6"))+
          theme(legend.text=element_text(size=18),
                legend.title=element_text(size=24))

ggsave(filename="plots/Encircled_Image_Histograms_NIH_PILL.png", plot=scat,
       width=9, height=7)

#setup for validation plot

valid_results<-matrix(nrow=3, ncol=4, data=0)
colnames(valid_results)<-c("n=3", "n=4", "n=5", "n=6")
rownames(valid_results)<-c("CNN", "SVM", "Tree")

#setup for training plot
train_results<-matrix(nrow=3, ncol=4, data=0)
colnames(train_results)<-c("n=3", "n=4", "n=5", "n=6")
rownames(train_results)<-c("CNN", "SVM", "Tree")


##################################
## training sample size = 3
##################################

n=3

#cnn results for n=1
model_rslts[1,]<-c(1.00, 0.54)


#################
# modeling
#################

#finding those observations to train and validate on

set.seed(83150)


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

    train3<-sample(1:12, n)
    train4<-sample(1:8,  n)
    train5<-sample(1:12, n)
    train6<-sample(1:8,  n)

    mytrain<-rbind(tris[train3,], squs[train4,],
                   rbind(pens, pens2)[train5,],
                   rbind(hexs, hexs2)[train6,])

    labs_train<-as.factor(c(rep(1, n), rep(2, n),
                            rep(3, n), rep(4, n) ) )


    myvalid<-rbind(tris[-train3,], squs[-train4,],
                  rbind(pens, pens2)[-train5,],
                  rbind(hexs, hexs2)[-train6,])

    labs_valid<-as.factor(c(rep(1, 12-n), rep(2, 8-n),
                            rep(3, 12-n), rep(4, 8-n) ) )

    #######
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="polynomial",
               cost=2, coef0= 1, degree=2,
               #cost=1, coef0= 1, degree=2, 70%
               scale=FALSE)

    #plot(svmfit , train)

    #summary(svmfit)

    ypred=predict(svmfit ,train)
    table(predict=ypred, truth=train$labs)
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
    tree_train<-mean(ypred_train==as.factor((train$labs)))

    #plot(treefit )
    #text(treefit ,pretty =0)

    ypred_valid=predict(treefit ,valid, type='class')
    #table(predict=ypred_valid, truth=valid$labs)
    tree_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}


#################
## Model Results
#################


#SVM
model_rslts[2,1]<-mean(svm_train)
model_rslts[2,2]<-mean(svm_valid)

#tree
model_rslts[3,1]<-mean(tree_train)
model_rslts[3,2]<-mean(tree_valid)

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

#cnn results for n=1
model_rslts[1,]<-c(1.00, 0.54)


#################
# modeling
#################

#finding those observations to train and validate on

set.seed(6204064)


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

    train3<-sample(1:12, n)
    train4<-sample(1:8,  n)
    train5<-sample(1:12, n)
    train6<-sample(1:8,  n)

    mytrain<-rbind(tris[train3,], squs[train4,],
                   rbind(pens, pens2)[train5,],
                   rbind(hexs, hexs2)[train6,])

    labs_train<-as.factor(c(rep(1, n), rep(2, n),
                            rep(3, n), rep(4, n) ) )


    myvalid<-rbind(tris[-train3,], squs[-train4,],
                  rbind(pens, pens2)[-train5,],
                  rbind(hexs, hexs2)[-train6,])

    #
    labs_valid<-as.factor(c(rep(1, 12-n), rep(2, 8-n),
                            rep(3, 12-n), rep(4, 8-n) ) )

    #######
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="polynomial",
               cost=2, coef0= 1, degree=2,
               #cost=1, coef0= 1, degree=2, 70%
               scale=FALSE)

    #plot(svmfit , train)

    #summary(svmfit)

    ypred=predict(svmfit ,train)
    table(predict=ypred, truth=train$labs)
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
    tree_train<-mean(ypred_train==as.factor((train$labs)))

    #plot(treefit )
    #text(treefit ,pretty =0)

    ypred_valid=predict(treefit ,valid, type='class')
    #table(predict=ypred_valid, truth=valid$labs)
    tree_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}


#################
## Model Results
#################

#SVM
model_rslts[2,1]<-mean(svm_train)
model_rslts[2,2]<-mean(svm_valid)

#tree
model_rslts[3,1]<-mean(tree_train)
model_rslts[3,2]<-mean(tree_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)


#display results
model_rslts

xtable(model_rslts, digits=2)

valid_results[,2]<-model_rslts[,2]
train_results[,2]<-model_rslts[,1]


##################################
## training sample size = 5
##################################

n=5

#cnn results for n=1
model_rslts[1,]<-c(1.00, 0.54)


#################
# modeling
#################

#finding those observations to train and validate on

set.seed(9197216)


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

    train3<-sample(1:12, n)
    train4<-sample(1:8,  n)
    train5<-sample(1:12, n)
    train6<-sample(1:8,  n)

    mytrain<-rbind(tris[train3,], squs[train4,],
                   rbind(pens, pens2)[train5,],
                   rbind(hexs, hexs2)[train6,])

    labs_train<-as.factor(c(rep(1, n), rep(2, n),
                            rep(3, n), rep(4, n) ) )


    myvalid<-rbind(tris[-train3,], squs[-train4,],
                  rbind(pens, pens2)[-train5,],
                  rbind(hexs, hexs2)[-train6,])

    #
    labs_valid<-as.factor(c(rep(1, 12-n), rep(2, 8-n),
                            rep(3, 12-n), rep(4, 8-n) ) )

    #######
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="polynomial",
               cost=2, coef0= 1, degree=2,
               #cost=1, coef0= 1, degree=2, 70%
               scale=FALSE)

    #plot(svmfit , train)

    #summary(svmfit)

    ypred=predict(svmfit ,train)
    table(predict=ypred, truth=train$labs)
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
    tree_train<-mean(ypred_train==as.factor((train$labs)))

    #plot(treefit )
    #text(treefit ,pretty =0)

    ypred_valid=predict(treefit ,valid, type='class')
    #table(predict=ypred_valid, truth=valid$labs)
    tree_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}


#################
## Model Results
#################

#SVM
model_rslts[2,1]<-mean(svm_train)
model_rslts[2,2]<-mean(svm_valid)

#tree
model_rslts[3,1]<-mean(tree_train)
model_rslts[3,2]<-mean(tree_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)


#display results
model_rslts

xtable(model_rslts, digits=2)

valid_results[,3]<-model_rslts[,2]
train_results[,3]<-model_rslts[,1]


##################################
## training sample size = 6
##################################

n=6

#cnn results for n=1
model_rslts[1,]<-c(1.00, 0.54)


#################
# modeling
#################

#finding those observations to train and validate on

set.seed(57275)


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

    train3<-sample(1:12, n)
    train4<-sample(1:8,  n)
    train5<-sample(1:12, n)
    train6<-sample(1:8,  n)

    mytrain<-rbind(tris[train3,], squs[train4,],
                   rbind(pens, pens2)[train5,],
                   rbind(hexs, hexs2)[train6,])

    labs_train<-as.factor(c(rep(1, n), rep(2, n),
                            rep(3, n), rep(4, n) ) )


    myvalid<-rbind(tris[-train3,], squs[-train4,],
                  rbind(pens, pens2)[-train5,],
                  rbind(hexs, hexs2)[-train6,])

    #
    labs_valid<-as.factor(c(rep(1, 12-n), rep(2, 8-n),
                            rep(3, 12-n), rep(4, 8-n) ) )

    #######
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="polynomial",
               cost=2, coef0= 1, degree=2,
               #cost=1, coef0= 1, degree=2, 70%
               scale=FALSE)

    #plot(svmfit , train)

    #summary(svmfit)

    ypred=predict(svmfit ,train)
    table(predict=ypred, truth=train$labs)
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
    tree_train<-mean(ypred_train==as.factor((train$labs)))

    #plot(treefit )
    #text(treefit ,pretty =0)

    ypred_valid=predict(treefit ,valid, type='class')
    #table(predict=ypred_valid, truth=valid$labs)
    tree_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))

}


#################
## Model Results
#################


#SVM
model_rslts[2,1]<-mean(svm_train)
model_rslts[2,2]<-mean(svm_valid)

#tree
model_rslts[3,1]<-mean(tree_train)
model_rslts[3,2]<-mean(tree_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)


#display results
model_rslts

xtable(model_rslts, digits=2)

valid_results[,4]<-model_rslts[,2]
train_results[,4]<-model_rslts[,1]


train_results

valid_results

xtable(valid_results)

xtable(train_results)

ultima<-as.data.frame(rbind(train_results, valid_results))

fcts<-as.factor(c(rep(1, 3), rep(2, 3)))

ultima<-cbind(ultima, fcts)

ultima

xtable(ultima)


#final results plot

models<-( rep(rownames(ultima)[1:3], 4*3 ) )
set<-( rep(c(rep("Training", 3), rep("Validation", 3)), 3) )
acc<-c(ultima[,1], ultima[,2], ultima[,3])
samp<-c( rep(3.0, 6), rep(4.0, 6), rep(5.0, 6), rep(6.0, 6))
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
	 	  ggtitle("Overall Results for\nNML NIH Pills")+
		  xlab("Training Size")+
		  ylab("Overall Accuracy")+
		  labs(colour= "Model", shape="Data Set", linetype="Data Set")+
	      #scale_y_discrete(limits=c(0, 1.00))+
          #scale_x_discrete(breaks=c(3, 4, 5, 7, 10, 20))+
          mytheme.scat+
          scale_colour_manual(values = c("Red", "Green", "khaki2"))+
          #scale_color_discrete(breaks=c("Training", "Validation"))+
          theme(legend.text=element_text(size=18),
                legend.title=element_text(size=24))

ultima_plot

ggsave(filename="plots/OverallAcc_nml.png", plot=ultima_plot,
       width=9, height=7)


##########################
# Empirical SP Estimation
##########################


mydata<-rbind(tris, squs, pens, hexs, hepts, octs)

labs<-as.factor(c(rep(1, dim(tris)[1]), rep(2, dim(squs)[1]),
                  rep(3, dim(pens)[1]), rep(4, dim(hexs)[1]),
                  rep(5, dim(hepts)[1]), rep(6, dim(octs)[1]) ) )


sps<-mydata[,1]/rowSums(mydata)
aggregate(sps~labs, FUN=mean)
xtable(aggregate(sps~labs, FUN=sd))




#
