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

#https://michael.hahsler.net/SMU/EMIS7332/R/viz_classifier.html
#decision plot code from above
decisionplot <- function(model, data, class = NULL, predict_type = "class",
  resolution = 100, showgrid = TRUE, ...) {

  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  k <- length(unique(cl))

  plot(data, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, ...)

  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)

  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)

  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")

  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
    lwd = 2, levels = (1:(k-1))+.5)

  invisible(z)
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


#cleaning data for ggplot2 and analysis
labs<-as.factor(c(rep(1, dim(tris)[1]), rep(2, dim(squs)[1]),
                  rep(3, dim(pens)[1]), rep(4, dim(hexs)[1]),
                  rep(5, dim(hepts)[1]), rep(6, dim(octs)[1]) ) )

mydata<-rbind(tris, squs, pens, hexs, hepts, octs)

#counts plot
temp<-as.data.frame(cbind(labs, mydata))
labs2<-as.factor(c(rep("n=3", dim(tris)[1]), rep("n=4", dim(squs)[1]), rep("n=5", dim(pens)[1]),
                rep("n=6", dim(hexs)[1]), rep("n=7", dim(hepts)[1]),   rep("n=8", dim(octs)[1]) ))


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


#setup for validation plot

valid_results<-matrix(nrow=4, ncol=6, data=0)
colnames(valid_results)<-c("n=3", "n=4", "n=5", "n=6", "n=7", "n=8")
rownames(valid_results)<-c("CNN", "QDA", "SVM", "Tree")

#setup for training plot
train_results<-matrix(nrow=4, ncol=6, data=0)
colnames(train_results)<-c("n=3", "n=4", "n=5", "n=6", "n=7", "n=8")
rownames(train_results)<-c("CNN", "QDA", "SVM", "Tree")


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
svm_train<-c()
svm_valid<-c()
tree_train<-c()
tree_valid<-c()

#simuiltion size

i=1

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
    #SVM
    #######

    train<-as.data.frame(cbind(as.factor(labs_train), mytrain))
    colnames(train)[1]<-"labs"

    valid<-as.data.frame(cbind(as.factor(labs_valid), myvalid))
    colnames(valid)[1]<-"labs"

    #creating model
    svmfit=svm(labs ~ white + black, data=train, kernel="linear", cost=1,
    scale=FALSE)

    viz_data<-cbind(mydata, labs2)
    colnames(viz_data)[3]<-"Number of Sides"
    par(oma=c(0,0,2,0))
    decisionplot(svmfit, viz_data, class = "Number of Sides",
                main = "",
                xlab='',
                ylab=''
                )
    title(cex.main=4,
          main = "SVM (linear)",
          cex.lab=1.5,
          xlab='White',
          ylab='Black'
          )

    #summary(svmfit)

    ypred=predict(svmfit ,train)
    #table(predict=ypred, truth=train$labs)
    svm_train[i]<-mean(ypred==as.factor(as.numeric(labs_train)))

    #now on valid
    ypred_valid=predict(svmfit ,valid)
    #table(predict=ypred_valid, truth=valid$labs)
    svm_valid[i]<-mean(ypred_valid==as.factor(as.numeric(labs_valid)))
    #model_rslts



#
