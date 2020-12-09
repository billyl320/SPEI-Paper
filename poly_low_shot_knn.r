#loading libraries
library('class')#for knn
library('xtable')#for LaTex write up


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
sp<-mydata[,1]/rowSums(mydata)
mydata<-cbind(mydata,sp)
colnames(mydata)[3]<-'sp'

#counts plot
temp<-as.data.frame(cbind(labs, mydata))
labs2<-as.factor(c(rep("n=3", dim(tris)[1]), rep("n=4", dim(squs)[1]), rep("n=5", dim(pens)[1]),
                rep("n=6", dim(hexs)[1]), rep("n=7", dim(hepts)[1]),   rep("n=8", dim(octs)[1]) ))


mydata<-cbind(labs, mydata)

#
#######################
#running experiment
########################

#set seed
set.seed(117018)

#accuracy object
acc<-c()

for(i in 1:1000){
  #use the 3 classes
  classes<-c(1, 2, 3)

  #find those 5 classes in mydata
  keep1<-which(labs==classes[1])
  keep2<-which(labs==classes[2])
  keep3<-which(labs==classes[3])


  #sample 20 images;
  #the first is the 'known' and the remainder are the 'unknown'
  obs1<-sample(keep1, 20)
  obs2<-sample(keep2, 20)
  obs3<-sample(keep3, 20)

  #setup known and unknown data objects
  known<-rbind(mydata[obs1[1],],mydata[obs2[1],],
               mydata[obs3[1],]
              )
  unknown<-rbind(mydata[obs1[-1],],mydata[obs2[-1],],
                 mydata[obs3[-1],]
                )

  knn_fit<-knn(train=as.matrix(known[,4]),
               test=as.matrix(unknown[,4]),
               cl=known[,1],
               k=1
               )
  tab <- table(prediction=knn_fit,truth=unknown[,1])
  acc[i]<-sum(diag(tab))/length(unknown[,1])
}

#using only SP 91%
mean(acc)

#quantile is 58% to 98%
quantile(acc, probs=c(0.025, 0.975))





#
