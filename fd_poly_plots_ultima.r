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


#importing data for encircled image histograms
tris <- read.table("tris_fd.txt", sep=",", header=TRUE)
squs <- read.table("squs_fd.txt", sep=",", header=TRUE)
pens <- read.table("pens_fd.txt", sep=",", header=TRUE)
hexs <- read.table("hexs_fd.txt", sep=",", header=TRUE)
hepts <- read.table("hepts_fd.txt", sep=",", header=TRUE)
octs <- read.table("octs_fd.txt", sep=",", header=TRUE)


#cleaning data for ggplot2 and analysis
labs<-as.factor(c(rep(1, dim(tris)[1]), rep(2, dim(squs)[1]),
                  rep(3, dim(pens)[1]), rep(4, dim(hexs)[1]),
                  rep(5, dim(hepts)[1]), rep(6, dim(octs)[1]) ) )

mydata<-rbind(tris, squs, pens, hexs, hepts, octs)

#counts plot
temp<-as.data.frame(cbind(labs, mydata))
labs2<-as.factor(c(rep("n=3", dim(tris)[1]), rep("n=4", dim(squs)[1]), rep("n=5", dim(pens)[1]),
                rep("n=6", dim(hexs)[1]), rep("n=7", dim(hepts)[1]),   rep("n=8", dim(octs)[1]) ))


scat<-ggplot(data=temp, aes(-1*fd_box, fill=as.factor(labs2)))+
          geom_histogram()+
	 	      ggtitle("FD via Box-Counting for\nCreated Polygons")+
		      xlab("FD")+
					ylab("Frequency")+
			 		labs(fill= "Legend")+
					scale_y_continuous(label=scientific_10)+
          scale_x_continuous(label=scientific_10)+
          mytheme.scat+
          scale_color_discrete(breaks=c("n=3","n=4","n=5", "n=6",
                                        "n=7", "n=8"))+
          theme(legend.text=element_text(size=18),
                legend.title=element_text(size=24))

ggsave(filename="plots/fd_regular_poly.png", plot=scat,
       width=10, height=8)

#
scat2<-ggplot(data=temp, aes(fd_min, fill=as.factor(labs2)))+
          geom_histogram()+
	 	      ggtitle("FD via Dilation for\nCreated Polygons")+
		      xlab("FD")+
					ylab("Frequency")+
			 		labs(fill= "Legend")+
					scale_y_continuous(label=scientific_10)+
          scale_x_continuous(label=scientific_10)+
          mytheme.scat+
          scale_color_discrete(breaks=c("n=3","n=4","n=5", "n=6",
                                        "n=7", "n=8"))+
          theme(legend.text=element_text(size=18),
                legend.title=element_text(size=24))

ggsave(filename="plots/fd_dilaiton_regular_poly.png", plot=scat2,
       width=10, height=8)

#
scat3<-ggplot(data=temp, aes(fd_mass, fill=as.factor(labs2)))+
          geom_histogram()+
	 	      ggtitle("FD via Mass-Radius for\nCreated Polygons")+
		      xlab("FD")+
					ylab("Frequency")+
			 		labs(fill= "Legend")+
					scale_y_continuous(label=scientific_10)+
          scale_x_continuous(label=scientific_10)+
          mytheme.scat+
          scale_color_discrete(breaks=c("n=3","n=4","n=5", "n=6",
                                        "n=7", "n=8"))+
          theme(legend.text=element_text(size=18),
                legend.title=element_text(size=24))

ggsave(filename="plots/fd_mass_regular_poly.png", plot=scat3,
       width=10, height=8)

#
