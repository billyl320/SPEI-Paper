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

model_rslts[1,]<-c(0.95, 0.60)

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

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(83150)

train3<-sample(1:12, 6)
train4<-sample(1:8,  4)
train5<-sample(1:12, 6)
train6<-sample(1:8,  4)

mytrain<-rbind(tris[train3,], squs[train4,],
               rbind(pens, pens2)[train5,],
               rbind(hexs, hexs2)[train6,])

labs_train<-as.factor(c(rep(1, 6), rep(2, 4),
                        rep(3, 6), rep(4, 4) ) )


myvalid<-rbind(tris[-train3,], squs[-train4,],
              rbind(pens, pens2)[-train5,],
              rbind(hexs, hexs2)[-train6,])

#
labs_valid<-as.factor(c(rep(1, 6), rep(2, 4),
                        rep(3, 6), rep(4, 4) ) )

#####

#######
#QDA
#######
train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
colnames(train)[1]<-"labs"


#creating model
qda.fit = qda(labs ~ white + black, data=train)
qda.fit #rank deficiency - ie unable to compute

#predicting
qda.pred=predict(qda.fit, train)
qda.class = qda.pred$class

#results
table(qda.class, labs_train)
#overall classification rate for training
model_rslts[2,1]<- mean(qda.class==as.factor(as.numeric(labs_train)))

####
#now predict on validation
valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
colnames(valid)[1]<-"labs"

#predicting
qda.pred=predict(qda.fit, valid)
qda.class = qda.pred$class

#results
table(qda.class, labs_valid)
#overall classification rate for training
model_rslts[2,2]<-mean(qda.class==as.factor(as.numeric(labs_valid)))

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

plot(svmfit , train)

summary(svmfit)

ypred=predict(svmfit ,train)
table(predict=ypred, truth=train$labs)
model_rslts[3,1]<-mean(ypred==as.factor(as.numeric(labs_train)))

#now on valid
ypred_valid=predict(svmfit ,valid)
table(predict=ypred_valid, truth=valid$labs)
model_rslts[3,2]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))
model_rslts


######
# Tree
#######

#training tree mdoel
treefit =tree(labs ~ white + black, data=train )
summary(treefit)

model_rslts[4,1]<-10 / 20

plot(treefit )
text(treefit ,pretty =0)

ypred_valid=predict(treefit ,valid, type='class')
table(predict=ypred_valid, truth=valid$labs)
model_rslts[4,2]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))
model_rslts

xtable(model_rslts, digits=2)



##########################
# Empirical SP Estimation
##########################

sps<-mydata[,1]/rowSums(mydata)
aggregate(sps~labs, FUN=mean)
xtable(aggregate(sps~labs, FUN=sd))




#
