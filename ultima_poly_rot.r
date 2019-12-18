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
model_rslts<-matrix(nrow=2, ncol=1, data=0)
colnames(model_rslts)<-c("Validation")
rownames(model_rslts)<-c("Original", "Rotated")

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

#scatterplots
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

valid_results<-matrix(nrow=2, ncol=6, data=0)
colnames(valid_results)<-c("n=3", "n=4", "n=5", "n=6", "n=7", "n=8")
rownames(valid_results)<-c("Original", "Rotated")

##################################
## training sample size = 3
##################################

n=3

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(695304)

#initialize objects to hold results
qda_train<-c()
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
    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"

    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid)[1]<-"labs"


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
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

}

#################
## Model Results
#################

#QDA
model_rslts[1,1]<-mean(qda_valid)
model_rslts[2,1]<-mean(rot_valid)


sd(qda_valid)
sd(qda_train)
sd(rot_valid)

#display results
model_rslts

xtable(model_rslts, digits=2)

##################################
## training sample size = 4
##################################

n=4

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(555665)

#initialize objects to hold results
qda_train<-c()
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

    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"

    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid)[1]<-"labs"


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
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

}

#################
## Model Results
#################

#QDA
model_rslts[1,1]<-mean(qda_valid)
model_rslts[2,1]<-mean(rot_valid)


sd(qda_valid)
sd(qda_train)
sd(rot_valid)

#display results
model_rslts

xtable(model_rslts, digits=2)


##################################
## training sample size = 5
##################################

n=5

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(723019)

#initialize objects to hold results
qda_train<-c()
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

    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"

    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid)[1]<-"labs"


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
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

}

#################
## Model Results
#################

#QDA
model_rslts[1,1]<-mean(qda_valid)
model_rslts[2,1]<-mean(rot_valid)


sd(qda_valid)
sd(qda_train)
sd(rot_valid)

#display results
model_rslts

xtable(model_rslts, digits=2)


##################################
## training sample size = 6
##################################

n=6

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(442644)

#initialize objects to hold results
qda_train<-c()
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
    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"

    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid)[1]<-"labs"


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
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

}

#################
## Model Results
#################

#QDA
model_rslts[1,1]<-mean(qda_valid)
model_rslts[2,1]<-mean(rot_valid)


sd(qda_valid)
sd(qda_train)
sd(rot_valid)

#display results
model_rslts

xtable(model_rslts, digits=2)

##################################
## training sample size = 7
##################################

n=7

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(459237)
#initialize objects to hold results
qda_train<-c()
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
    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"

    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid)[1]<-"labs"


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
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

}

#################
## Model Results
#################

#QDA
model_rslts[1,1]<-mean(qda_valid)
model_rslts[2,1]<-mean(rot_valid)


sd(qda_valid)
sd(qda_train)
sd(rot_valid)

#display results
model_rslts

xtable(model_rslts, digits=2)


##################################
## training sample size = 8
##################################

n=8

#################
# modeling
#################

#finding those observations to train and validate on

set.seed(326668)
#initialize objects to hold results
qda_train<-c()
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
    #######
    #QDA
    #######
    temp<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(temp)[1]<-"labs"

    valid_rot<-as.data.frame(cbind(as.factor(labs_rot), mydata_rot))
    colnames(valid)[1]<-"labs"


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
    qda.pred=predict(qda.fit, valid_rot)
    qda.class = qda.pred$class
    #table(predict=ypred_valid, truth=valid$labs)
    rot_valid[i]<-mean(qda.class==as.factor(as.numeric(labs_rot)))

}

#################
## Model Results
#################

#QDA
model_rslts[1,1]<-mean(qda_valid)
model_rslts[2,1]<-mean(rot_valid)


sd(qda_valid)
sd(qda_train)
sd(rot_valid)

#display results
model_rslts

xtable(model_rslts, digits=2)