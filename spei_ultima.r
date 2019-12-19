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
model_rslts<-matrix(nrow=4, ncol=2, data=0)
colnames(model_rslts)<-c("Train", "Validation")
rownames(model_rslts)<-c("CNN", "QDA", "SVM", "Tree")

model_rslts[1,]<-c(0.95, 0.27)

#importing data for encircled image histograms
tris <- read.table("tris.txt", sep=",", header=TRUE)
squs <- read.table("squs.txt", sep=",", header=TRUE)
pens <- read.table("pens.txt", sep=",", header=TRUE)
hexs <- read.table("hexs.txt", sep=",", header=TRUE)
hepts <- read.table("hepts.txt", sep=",", header=TRUE)
octs <- read.table("octs.txt", sep=",", header=TRUE)

#rotated
tris_rot <- read.table("tris_rot.txt", sep=",", header=TRUE)
squs_rot <- read.table("squs_rot.txt", sep=",", header=TRUE)
pens_rot <- read.table("pens_rot.txt", sep=",", header=TRUE)
hexs_rot <- read.table("hexs_rot.txt", sep=",", header=TRUE)
hepts_rot <- read.table("hept_rot.txt", sep=",", header=TRUE)
octs_rot <- read.table("octs_rot.txt", sep=",", header=TRUE)


#cleaning data for ggplot2 and analysis
labs<-as.factor(c(rep(1, dim(tris)[1]), rep(2, dim(squs)[1]),
                  rep(3, dim(pens)[1]), rep(4, dim(hexs)[1]),
                  rep(5, dim(hepts)[1]), rep(6, dim(octs)[1]) ) )

mydata<-rbind(tris, squs, pens, hexs, hepts, octs)

#counts plot
temp<-as.data.frame(cbind(labs, mydata))
labs2<-as.factor(c(rep("n=3", dim(tris)[1]), rep("n=4", dim(squs)[1]), rep("n=5", dim(pens)[1]),
                rep("n=6", dim(hexs)[1]), rep("n=7", dim(hepts)[1]),   rep("n=8", dim(octs)[1]) ))

#rotated
labs_rot<-as.factor(c(rep(1, dim(tris_rot)[1]), rep(2, dim(squs_rot)[1]),
                  rep(3, dim(pens_rot)[1]), rep(4, dim(hexs_rot)[1]),
                  rep(5, dim(hepts_rot)[1]), rep(6, dim(octs_rot)[1]) ) )

mydata_rot<-rbind(tris_rot, squs_rot, pens_rot, hexs_rot, hepts_rot, octs_rot)

#counts plot
temp_rot<-as.data.frame(cbind(labs_rot, mydata_rot))
labs2_rot<-as.factor(c(rep("n=3", dim(tris_rot)[1]), rep("n=4", dim(squs_rot)[1]), rep("n=5", dim(pens_rot)[1]),
                rep("n=6", dim(hexs_rot)[1]), rep("n=7", dim(hepts_rot)[1]),   rep("n=8", dim(octs_rot)[1]) ))

scat<-ggplot(data=temp, aes(x = white, y = black, colour = as.factor(labs2)))+
          geom_point(size=2)+
	 	      ggtitle("EI for\nCreated Polygons")+
		      xlab("White Counts")+
					ylab("Black Counts")+
			 		labs(colour= "Legend")+
					scale_y_continuous(label=scientific_10)+
          scale_x_continuous(label=scientific_10)+
          mytheme.scat+
          scale_color_discrete(breaks=c("n=3","n=4","n=5", "n=6",
                                        "n=7", "n=8"))+
          theme(legend.text=element_text(size=18),
                legend.title=element_text(size=24))

ggsave(filename="plots/Encircled_Image_Histograms.png", plot=scat,
       width=9, height=7)

# Rotated
scat<-ggplot(data=temp_rot, aes(x = white, y = black, colour = as.factor(labs2_rot)))+
          geom_point(size=2)+
	 	      ggtitle("EI for\nRotated Polygons")+
		      xlab("White Counts")+
					ylab("Black Counts")+
			 		labs(colour= "Legend")+
					scale_y_continuous(label=scientific_10)+
          scale_x_continuous(label=scientific_10)+
          mytheme.scat+
          scale_color_discrete(breaks=c("n=3","n=4","n=5", "n=6",
                                        "n=7", "n=8"))+
          theme(legend.text=element_text(size=18),
                legend.title=element_text(size=24))

ggsave(filename="plots/Encircled_Image_Histograms_Rotated.png", plot=scat,
       width=9, height=7)

#setup for validation plot

valid_results<-matrix(nrow=4, ncol=6, data=0)
colnames(valid_results)<-c("n=3", "n=4", "n=5", "n=6", "n=7", "n=8")
rownames(valid_results)<-c("CNN", "QDA", "SVM", "Tree")

#setup for training plot
train_results<-matrix(nrow=4, ncol=6, data=0)
colnames(train_results)<-c("n=3", "n=4", "n=5", "n=6", "n=7", "n=8")
rownames(train_results)<-c("CNN", "QDA", "SVM", "Tree")

##################################
## training sample size = 3
##################################

n=3

#cnn results for n=1
model_rslts[1,]<-c(0.95, 0.27)


#################
# modeling
#################

#finding those observations to train and validate on

set.seed(695304)

#initialize objects to hold results
qda_train<-c()
qda_valid<-c()
svm_train<-c()
svm_valid<-c()
tree_train<-c()
tree_valid<-c()
qda_valid<-c()
rot_valid<-c()

#simuiltion size
sim=100

for (i in 1:sim) {

    train3<-sample(1:125, n)
    train4<-sample(1:125, n)
    train5<-sample(1:125, n)
    train6<-sample(1:125, n)
    train7<-sample(1:125, n)
    train8<-sample(1:125, n)

    mytrain<-rbind(tris[train3,], squs[train4,], pens[train5,],
                   hexs[train6,], hepts[train7,], octs[train8,])

    labs_train<-as.factor(c(rep(1, n), rep(2, n),
                      rep(3, n), rep(4, n),
                      rep(5, n), rep(6, n) ) )


    myvalid<-rbind(tris[-train3,], squs[-train4,], pens[-train5,],
                   hexs[-train6,], hepts[-train7,], octs[-train8,])

    labs_valid<-as.factor(c(rep(1, 125-n), rep(2, 125-n),
                      rep(3, 125-n), rep(4, 125-n),
                      rep(5, 125-n), rep(6, 125-n) ) )

    #####

    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"


    #creating model
    qda.fit = qda(labs ~ white + black, data=temp)
    #qda.fit #rank deficiency - ie unable to compute

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_train)
    #overall classification rate for training
    qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

    ####
    #now predict on validation
    temp<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(temp)[1]<-"labs"

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_valid)
    #overall classification rate for training
    qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

    #rotated results
    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid_rot)[1]<-"labs"
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

    #######
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="linear", cost=1,
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
    #model_rslts


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

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

#rotated
mean(rot_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)
sd(rot_valid)

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
model_rslts[1,]<-c(0.91, 0.30)


#################
# modeling
#################

#finding those observations to train and validate on

set.seed(555665)

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

    train3<-sample(1:125, n)
    train4<-sample(1:125, n)
    train5<-sample(1:125, n)
    train6<-sample(1:125, n)
    train7<-sample(1:125, n)
    train8<-sample(1:125, n)

    mytrain<-rbind(tris[train3,], squs[train4,], pens[train5,],
                   hexs[train6,], hepts[train7,], octs[train8,])

    labs_train<-as.factor(c(rep(1, n), rep(2, n),
                      rep(3, n), rep(4, n),
                      rep(5, n), rep(6, n) ) )


    myvalid<-rbind(tris[-train3,], squs[-train4,], pens[-train5,],
                   hexs[-train6,], hepts[-train7,], octs[-train8,])

    labs_valid<-as.factor(c(rep(1, 125-n), rep(2, 125-n),
                      rep(3, 125-n), rep(4, 125-n),
                      rep(5, 125-n), rep(6, 125-n) ) )

    #####

    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"


    #creating model
    qda.fit = qda(labs ~ white + black, data=temp)
    #qda.fit #rank deficiency - ie unable to compute

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_train)
    #overall classification rate for training
    qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

    ####
    #now predict on validation
    temp<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(temp)[1]<-"labs"

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_valid)
    #overall classification rate for training
    qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

    #rotated results
    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid_rot)[1]<-"labs"
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

    #######
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="linear", cost=1,
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
    #model_rslts


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

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

#rotated
mean(rot_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)
sd(rot_valid)

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
model_rslts[1,]<-c(0.86, 0.32)


#################
# modeling
#################

#finding those observations to train and validate on

set.seed(723019)

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

    train3<-sample(1:125, n)
    train4<-sample(1:125, n)
    train5<-sample(1:125, n)
    train6<-sample(1:125, n)
    train7<-sample(1:125, n)
    train8<-sample(1:125, n)

    mytrain<-rbind(tris[train3,], squs[train4,], pens[train5,],
                   hexs[train6,], hepts[train7,], octs[train8,])

    labs_train<-as.factor(c(rep(1, n), rep(2, n),
                      rep(3, n), rep(4, n),
                      rep(5, n), rep(6, n) ) )


    myvalid<-rbind(tris[-train3,], squs[-train4,], pens[-train5,],
                   hexs[-train6,], hepts[-train7,], octs[-train8,])

    labs_valid<-as.factor(c(rep(1, 125-n), rep(2, 125-n),
                      rep(3, 125-n), rep(4, 125-n),
                      rep(5, 125-n), rep(6, 125-n) ) )

    #####

    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"


    #creating model
    qda.fit = qda(labs ~ white + black, data=temp)
    #qda.fit #rank deficiency - ie unable to compute

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_train)
    #overall classification rate for training
    qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

    ####
    #now predict on validation
    temp<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(temp)[1]<-"labs"

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_valid)
    #overall classification rate for training
    qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

    #rotated results
    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid_rot)[1]<-"labs"
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

    #######
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="linear", cost=1,
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
    #model_rslts


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

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

#rotated
mean(rot_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)
sd(rot_valid)

#display results
model_rslts

xtable(model_rslts, digits=2)

valid_results[,3]<-model_rslts[,2]
train_results[,3]<-model_rslts[,1]



##################################
## training sample size = 6
##################################

n=6

#cnn results for n=20
model_rslts[1,]<-c(0.92, 0.37)


#################
# modeling
#################

#finding those observations to train and validate on

set.seed(442644)

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

    train3<-sample(1:125, n)
    train4<-sample(1:125, n)
    train5<-sample(1:125, n)
    train6<-sample(1:125, n)
    train7<-sample(1:125, n)
    train8<-sample(1:125, n)

    mytrain<-rbind(tris[train3,], squs[train4,], pens[train5,],
                   hexs[train6,], hepts[train7,], octs[train8,])

    labs_train<-as.factor(c(rep(1, n), rep(2, n),
                      rep(3, n), rep(4, n),
                      rep(5, n), rep(6, n) ) )


    myvalid<-rbind(tris[-train3,], squs[-train4,], pens[-train5,],
                   hexs[-train6,], hepts[-train7,], octs[-train8,])

    labs_valid<-as.factor(c(rep(1, 125-n), rep(2, 125-n),
                      rep(3, 125-n), rep(4, 125-n),
                      rep(5, 125-n), rep(6, 125-n) ) )

    #####

    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"


    #creating model
    qda.fit = qda(labs ~ white + black, data=temp)
    #qda.fit #rank deficiency - ie unable to compute

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_train)
    #overall classification rate for training
    qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

    ####
    #now predict on validation
    temp<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(temp)[1]<-"labs"

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_valid)
    #overall classification rate for training
    qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

    #rotated results
    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid_rot)[1]<-"labs"
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

    #######
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="linear", cost=1,
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
    #model_rslts


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

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

#rotated
mean(rot_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)
sd(rot_valid)

#display results
model_rslts

xtable(model_rslts, digits=2)

valid_results[,4]<-model_rslts[,2]
train_results[,4]<-model_rslts[,1]



##################################
## training sample size = 7
##################################

n=7

#cnn results for n=25
model_rslts[1,]<-c(0.94, 0.42)


#################
# modeling
#################

#finding those observations to train and validate on

set.seed(459237)

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

    train3<-sample(1:125, n)
    train4<-sample(1:125, n)
    train5<-sample(1:125, n)
    train6<-sample(1:125, n)
    train7<-sample(1:125, n)
    train8<-sample(1:125, n)

    mytrain<-rbind(tris[train3,], squs[train4,], pens[train5,],
                   hexs[train6,], hepts[train7,], octs[train8,])

    labs_train<-as.factor(c(rep(1, n), rep(2, n),
                      rep(3, n), rep(4, n),
                      rep(5, n), rep(6, n) ) )


    myvalid<-rbind(tris[-train3,], squs[-train4,], pens[-train5,],
                   hexs[-train6,], hepts[-train7,], octs[-train8,])

    labs_valid<-as.factor(c(rep(1, 125-n), rep(2, 125-n),
                      rep(3, 125-n), rep(4, 125-n),
                      rep(5, 125-n), rep(6, 125-n) ) )

    #####

    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"


    #creating model
    qda.fit = qda(labs ~ white + black, data=temp)
    #qda.fit #rank deficiency - ie unable to compute

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_train)
    #overall classification rate for training
    qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

    ####
    #now predict on validation
    temp<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(temp)[1]<-"labs"

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_valid)
    #overall classification rate for training
    qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

    #rotated results
    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid_rot)[1]<-"labs"
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

    #######
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="linear", cost=1,
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
    #model_rslts


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

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

#rotated
mean(rot_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)
sd(rot_valid)

#display results
model_rslts

xtable(model_rslts, digits=2)

valid_results[,5]<-model_rslts[,2]
train_results[,5]<-model_rslts[,1]


##################################
## training sample size = 8
##################################

n=8

#cnn results for n=1
model_rslts[1,]<-c(0.92, 0.46)


#################
# modeling
#################

#finding those observations to train and validate on

set.seed(326668)

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

    train3<-sample(1:125, n)
    train4<-sample(1:125, n)
    train5<-sample(1:125, n)
    train6<-sample(1:125, n)
    train7<-sample(1:125, n)
    train8<-sample(1:125, n)

    mytrain<-rbind(tris[train3,], squs[train4,], pens[train5,],
                   hexs[train6,], hepts[train7,], octs[train8,])

    labs_train<-as.factor(c(rep(1, n), rep(2, n),
                      rep(3, n), rep(4, n),
                      rep(5, n), rep(6, n) ) )


    myvalid<-rbind(tris[-train3,], squs[-train4,], pens[-train5,],
                   hexs[-train6,], hepts[-train7,], octs[-train8,])

    labs_valid<-as.factor(c(rep(1, 125-n), rep(2, 125-n),
                      rep(3, 125-n), rep(4, 125-n),
                      rep(5, 125-n), rep(6, 125-n) ) )

    #####

    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"


    #creating model
    qda.fit = qda(labs ~ white + black, data=temp)
    #qda.fit #rank deficiency - ie unable to compute

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_train)
    #overall classification rate for training
    qda_train[i]<- mean(qda.class==as.factor(as.numeric(labs_train)))

    ####
    #now predict on validation
    temp<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(temp)[1]<-"labs"

    #predicting
    qda.pred=predict(qda.fit, temp)
    qda.class = qda.pred$class

    #results
    #table(qda.class, labs_valid)
    #overall classification rate for training
    qda_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

    #rotated results
    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid_rot)[1]<-"labs"
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

    #######
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="linear", cost=1,
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
    #model_rslts


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

#QDA
model_rslts[2,1]<-mean(qda_train)
model_rslts[2,2]<-mean(qda_valid)

#SVM
model_rslts[3,1]<-mean(svm_train)
model_rslts[3,2]<-mean(svm_valid)

#tree
model_rslts[4,1]<-mean(tree_train)
model_rslts[4,2]<-mean(tree_valid)

#rotated
mean(rot_valid)

sd(qda_train)
sd(qda_valid)
sd(svm_valid)
sd(svm_train)
sd(tree_train)
sd(tree_valid)
sd(rot_valid)

#display results
model_rslts

xtable(model_rslts, digits=2)

valid_results[,6]<-model_rslts[,2]
train_results[,6]<-model_rslts[,1]


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

models<-( rep(c("CNN","QDA", "SVM", "Tree" ), 12 ) )
set<-( rep(c(rep("Training", 4), rep("Validation", 4)), 6) )
acc<-c(ultima[,1], ultima[,2], ultima[,3],
       ultima[,4], ultima[,5], ultima[,6])
samp<-c( rep(3.0, 8), rep(4.0, 8),rep(5.0, 8),
         rep(6.0, 8), rep(7.0, 8), rep(8.0, 8))
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
	 	  ggtitle("Overall Results for\nCreated Polygons")+
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

ggsave(filename="plots/OverallAcc_poly.png", plot=ultima_plot,
       width=9, height=7)


##########################
# Empirical SP Estimation
##########################

mydata2<-rbind(tris, squs, pens, hexs, hepts, octs)

labs<-as.factor(c(rep(1, dim(tris)[1]), rep(2, dim(squs)[1]),
                  rep(3, dim(pens)[1]), rep(4, dim(hexs)[1]),
                  rep(5, dim(hepts)[1]), rep(6, dim(octs)[1]) ) )

sps<-mydata2[,1]/rowSums(mydata2)
aggregate(sps~labs, FUN=mean)
xtable(aggregate(sps~labs, FUN=sd))

mydata3<-rbind(tris_rot, squs_rot, pens_rot, hexs_rot, hepts_rot, octs_rot)

labs<-as.factor(c(rep(1, dim(tris_rot)[1]), rep(2, dim(squs_rot)[1]),
                  rep(3, dim(pens_rot)[1]), rep(4, dim(hexs_rot)[1]),
                  rep(5, dim(hepts_rot)[1]), rep(6, dim(octs_rot)[1]) ) )

sps<-mydata3[,1]/rowSums(mydata3)
aggregate(sps~labs_rot, FUN=mean)
xtable(aggregate(sps~labs_rot, FUN=sd))
