#clean workspace
rm(list=ls())
#loading libraries
library('class')#for knn
library('xtable')#for LaTex write up
library('nortest')#for anderson-darling test

#report session info
sessionInfo()

#standard error function

se <- function(x){

  sqrt(var(x)/length(x))

}

#custom fuction to extract classes from name
#reference: https://stackoverflow.com/questions/7963898/extracting-the-last-n-characters-from-a-string-in-r
substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

#######################
#reading in the data
#######################

#angelic
ang<-read.table('txt/angelic.txt', sep=',', header=TRUE)
ang_shape<-read.table('txt/angelic_SHAPES.txt', sep=',', header=TRUE)
ang<-cbind(ang, ang[,1]/sum(ang))
colnames(ang)[3]<-"sp"

#import names and have correct number for data augmentation
ang_nams<-rep(as.character(read.table('txt/NAMES_angelic.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(ang_nams, each=4)
ang_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
ang<-cbind(as.numeric(ang_class), ang, ang_shape)
colnames(ang)[1]<-"Class"
ang<-ang[order(ang$Class),]

#Atemayar_Qelisayer
ate<-read.table('txt/Atemayar_Qelisayer.txt', sep=',', header=TRUE)
ate_shape<-read.table('txt/Atemayar_Qelisayer_SHAPES.txt', sep=',', header=TRUE)
ate<-cbind(ate, ate[,1]/sum(ate))
colnames(ate)[3]<-"sp"

#import names and have correct number for data augmentation
ate_nams<-rep(as.character(read.table('txt/NAMES_Atemayar_Qelisayer.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(ate_nams, each=4)
ate_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
ate<-cbind(as.numeric(ate_class), ate, ate_shape)
colnames(ate)[1]<-"Class"
ate<-ate[order(ate$Class),]

#Atlantean
atl<-read.table('txt/Atlantean.txt', sep=',', header=TRUE)
atl_shape<-read.table('txt/Atlantean_SHAPES.txt', sep=',', header=TRUE)
atl<-cbind(atl, atl[,1]/sum(atl))
colnames(atl)[3]<-"sp"

#import names and have correct number for data augmentation
atl_nams<-rep(as.character(read.table('txt/NAMES_Atlantean.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(atl_nams, each=4)
atl_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
atl<-cbind(as.numeric(atl_class), atl, atl_shape)
colnames(atl)[1]<-"Class"
atl<-atl[order(atl$Class),]

#Aurek_Besh
aur<-read.table('txt/Aurek_Besh.txt', sep=',', header=TRUE)
aur_shape<-read.table('txt/Aurek_Besh_SHAPES.txt', sep=',', header=TRUE)
aur<-cbind(aur, aur[,1]/sum(aur))
colnames(aur)[3]<-"sp"

#import names and have correct number for data augmentation
aur_nams<-rep(as.character(read.table('txt/NAMES_Aurek_Besh.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(aur_nams, each=4)
aur_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
aur<-cbind(as.numeric(aur_class), aur, aur_shape)
colnames(aur)[1]<-"Class"
aur<-aur[order(aur$Class),]

#Avesta
ave<-read.table('txt/Avesta.txt', sep=',', header=TRUE)
ave_shape<-read.table('txt/Avesta_SHAPES.txt', sep=',', header=TRUE)
ave<-cbind(ave, ave[,1]/sum(ave))
colnames(ave)[3]<-"sp"

#import names and have correct number for data augmentation
ave_nams<-rep(as.character(read.table('txt/NAMES_Avesta.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(ave_nams, each=4)
ave_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
ave<-cbind(as.numeric(ave_class), ave, ave_shape)
colnames(ave)[1]<-"Class"
ave<-ave[order(ave$Class),]

#Cyrillic
cyr<-read.table('txt/Cyrillic.txt', sep=',', header=TRUE)
cyr_shape<-read.table('txt/Cyrillic_SHAPES.txt', sep=',', header=TRUE)
cyr<-cbind(cyr, cyr[,1]/sum(cyr))
colnames(cyr)[3]<-"sp"

#import names and have correct number for data augmentation
cyr_nams<-rep(as.character(read.table('txt/NAMES_Cyrillic.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(cyr_nams, each=4)
cyr_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
cyr<-cbind(as.numeric(cyr_class), cyr, cyr_shape)
colnames(cyr)[1]<-"Class"
cyr<-cyr[order(cyr$Class),]

#Ge_ez
gee<-read.table('txt/Ge_ez.txt', sep=',', header=TRUE)
gee_shape<-read.table('txt/Ge_ez_SHAPES.txt', sep=',', header=TRUE)
gee<-cbind(gee, gee[,1]/sum(gee))
colnames(gee)[3]<-"sp"

#import names and have correct number for data augmentation
gee_nams<-rep(as.character(read.table('txt/NAMES_Ge_ez.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(gee_nams, each=4)
gee_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
gee<-cbind(as.numeric(gee_class), gee, gee_shape)
colnames(gee)[1]<-"Class"
gee<-gee[order(gee$Class),]

#Glagolitic
gla<-read.table('txt/Glagolitic.txt', sep=',', header=TRUE)
gla_shape<-read.table('txt/Glagolitic_SHAPES.txt', sep=',', header=TRUE)
gla<-cbind(gla, gla[,1]/sum(gla))
colnames(gla)[3]<-"sp"

#import names and have correct number for data augmentation
gla_nams<-rep(as.character(read.table('txt/NAMES_Glagolitic.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(gla_nams, each=4)
gla_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
gla<-cbind(as.numeric(gla_class), gla, gla_shape)
colnames(gla)[1]<-"Class"
gla<-gla[order(gla$Class),]

#Gurmukhi
gur<-read.table('txt/Gurmukhi.txt', sep=',', header=TRUE)
gur_shape<-read.table('txt/Gurmukhi_SHAPES.txt', sep=',', header=TRUE)
gur<-cbind(gur, gur[,1]/sum(gur))
colnames(gur)[3]<-"sp"

#import names and have correct number for data augmentation
gur_nams<-rep(as.character(read.table('txt/NAMES_Gurmukhi.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(gur_nams, each=4)
gur_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
gur<-cbind(as.numeric(gur_class), gur, gur_shape)
colnames(gur)[1]<-"Class"
gur<-gur[order(gur$Class),]

#Kannada
kan<-read.table('txt/Kannada.txt', sep=',', header=TRUE)
kan_shape<-read.table('txt/Kannada_SHAPES.txt', sep=',', header=TRUE)
kan<-cbind(kan, kan[,1]/sum(kan))
colnames(kan)[3]<-"sp"

#import names and have correct number for data augmentation
kan_nams<-rep(as.character(read.table('txt/NAMES_Kannada.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(kan_nams, each=4)
kan_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
kan<-cbind(as.numeric(kan_class), kan, kan_shape)
colnames(kan)[1]<-"Class"
kan<-kan[order(kan$Class),]

#Keble
keb<-read.table('txt/Keble.txt', sep=',', header=TRUE)
keb_shape<-read.table('txt/Keble_SHAPES.txt', sep=',', header=TRUE)
keb<-cbind(keb, keb[,1]/sum(keb))
colnames(keb)[3]<-"sp"

#import names and have correct number for data augmentation
keb_nams<-rep(as.character(read.table('txt/NAMES_Keble.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(keb_nams, each=4)
keb_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
keb<-cbind(as.numeric(keb_class), keb, keb_shape)
colnames(keb)[1]<-"Class"
keb<-keb[order(keb$Class),]

#Malayalam
mal<-read.table('txt/Malayalam.txt', sep=',', header=TRUE)
mal_shape<-read.table('txt/Malayalam_SHAPES.txt', sep=',', header=TRUE)
mal<-cbind(mal, mal[,1]/sum(mal))
colnames(mal)[3]<-"sp"

#import names and have correct number for data augmentation
mal_nams<-rep(as.character(read.table('txt/NAMES_Malayalam.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(mal_nams, each=4)
mal_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
mal<-cbind(as.numeric(mal_class), mal, mal_shape)
colnames(mal)[1]<-"Class"
mal<-mal[order(mal$Class),]

#Manipuri
man<-read.table('txt/Manipuri.txt', sep=',', header=TRUE)
man_shape<-read.table('txt/Manipuri_SHAPES.txt', sep=',', header=TRUE)
man<-cbind(man, man[,1]/sum(man))
colnames(man)[3]<-"sp"

#import names and have correct number for data augmentation
man_nams<-rep(as.character(read.table('txt/NAMES_Manipuri.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(man_nams, each=4)
man_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
man<-cbind(as.numeric(man_class), man, man_shape)
colnames(man)[1]<-"Class"
man<-man[order(man$Class),]

#Mongolian
mon<-read.table('txt/Mongolian.txt', sep=',', header=TRUE)
mon_shape<-read.table('txt/Mongolian_SHAPES.txt', sep=',', header=TRUE)
mon<-cbind(mon, mon[,1]/sum(mon))
colnames(mon)[3]<-"sp"

#import names and have correct number for data augmentation
mon_nams<-rep(as.character(read.table('txt/NAMES_Mongolian.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(mon_nams, each=4)
mon_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
mon<-cbind(as.numeric(mon_class), mon, mon_shape)
colnames(mon)[1]<-"Class"
mon<-mon[order(mon$Class),]

#Oriya
ori<-read.table('txt/Oriya.txt', sep=',', header=TRUE)
ori_shape<-read.table('txt/Oriya_SHAPES.txt', sep=',', header=TRUE)
ori<-cbind(ori, ori[,1]/sum(ori))
colnames(ori)[3]<-"sp"

#import names and have correct number for data augmentation
ori_nams<-rep(as.character(read.table('txt/NAMES_Oriya.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(ori_nams, each=4)
ori_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
ori<-cbind(as.numeric(ori_class), ori, ori_shape)
colnames(ori)[1]<-"Class"
ori<-ori[order(ori$Class),]

#Serto
ser<-read.table('txt/Serto.txt', sep=',', header=TRUE)
ser_shape<-read.table('txt/Serto_SHAPES.txt', sep=',', header=TRUE)
ser<-cbind(ser, ser[,1]/sum(ser))
colnames(ser)[3]<-"sp"

#import names and have correct number for data augmentation
ser_nams<-rep(as.character(read.table('txt/NAMES_Serto.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(ser_nams, each=4)
ser_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
ser<-cbind(as.numeric(ser_class), ser, ser_shape)
colnames(ser)[1]<-"Class"
ser<-ser[order(ser$Class),]

#Sylheti
syl<-read.table('txt/Sylheti.txt', sep=',', header=TRUE)
syl_shape<-read.table('txt/Sylheti_SHAPES.txt', sep=',', header=TRUE)
syl<-cbind(syl, syl[,1]/sum(syl))
colnames(syl)[3]<-"sp"

#import names and have correct number for data augmentation
syl_nams<-rep(as.character(read.table('txt/NAMES_Sylheti.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(syl_nams, each=4)
syl_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
syl<-cbind(as.numeric(syl_class), syl, syl_shape)
colnames(syl)[1]<-"Class"
syl<-syl[order(syl$Class),]

#Tengwar
ten<-read.table('txt/Tengwar.txt', sep=',', header=TRUE)
ten_shape<-read.table('txt/Tengwar_SHAPES.txt', sep=',', header=TRUE)
ten<-cbind(ten, ten[,1]/sum(ten))
colnames(ten)[3]<-"sp"

#import names and have correct number for data augmentation
ten_nams<-rep(as.character(read.table('txt/NAMES_Tengwar.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(ten_nams, each=4)
ten_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
ten<-cbind(as.numeric(ten_class), ten, ten_shape)
colnames(ten)[1]<-"Class"
ten<-ten[order(ten$Class),]

#Tibetan
tib<-read.table('txt/Tibetan.txt', sep=',', header=TRUE)
tib_shape<-read.table('txt/Tibetan_SHAPES.txt', sep=',', header=TRUE)
tib<-cbind(tib, tib[,1]/sum(tib))
colnames(tib)[3]<-"sp"

#import names and have correct number for data augmentation
tib_nams<-rep(as.character(read.table('txt/NAMES_Tibetan.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(tib_nams, each=4)
tib_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
tib<-cbind(as.numeric(tib_class), tib, tib_shape)
colnames(tib)[1]<-"Class"
tib<-tib[order(tib$Class),]

#ULOG
ulo<-read.table('txt/ULOG.txt', sep=',', header=TRUE)
ulo_shape<-read.table('txt/ULOG_SHAPES.txt', sep=',', header=TRUE)
ulo<-cbind(ulo, ulo[,1]/sum(ulo))
colnames(ulo)[3]<-"sp"

#import names and have correct number for data augmentation
ulo_nams<-rep(as.character(read.table('txt/NAMES_ULOG.txt', sep='', header=TRUE)[,1]),each=4)
temp<-rep(ulo_nams, each=4)
ulo_class<-substr(substrRight(temp, 11), 1, 4)

#combine all of the objects into one
ulo<-cbind(as.numeric(ulo_class), ulo, ulo_shape)
colnames(ulo)[1]<-"Class"
ulo<-ulo[order(ulo$Class),]

#######################
#combine all of the
#objects into one large
#matrix
########################

mydata<-rbind(ang, ate, atl,
              aur, ave, cyr,
              gee, gla, gur,
              kan, keb, mal,
              man, mon, ori,
              ser, syl, ten,
              tib, ulo
             )

#
#######################
#running experiment
########################

#set seed
set.seed(4194693)


#number of episodes
num_ep<-10

#number of test episodes
num_test_ep<-1000

#accuracy object
acc<-c()
acc_info<-matrix(nrow=num_ep, ncol=3, data=0)
colnames(acc_info)<-c('Mean Acc.', '2.5% Quantile', '97.5 Quantile')
#mean accuracy object
mean_acc<-c()

for(j in 1:num_ep){

  for(i in 1:num_test_ep){
    #sample the five classes
    classes<-sample(unique(mydata$Class), 5)

    #find those 5 classes in mydata
    keep1<-which(mydata$Class==classes[1])
    keep2<-which(mydata$Class==classes[2])
    keep3<-which(mydata$Class==classes[3])
    keep4<-which(mydata$Class==classes[4])
    keep5<-which(mydata$Class==classes[5])

    #sample 20 images; the first is the 'known' and the remainder are the 'unknown'
    obs1<-sample(keep1, 20)
    obs2<-sample(keep2, 20)
    obs3<-sample(keep3, 20)
    obs4<-sample(keep4, 20)
    obs5<-sample(keep5, 20)

    #setup known and unknown data objects
    known<-rbind(mydata[obs1[1],],mydata[obs2[1],],
                 mydata[obs3[1],],mydata[obs4[1],],
                 mydata[obs5[1],]
                )
    unknown<-rbind(mydata[obs1[-1],],mydata[obs2[-1],],
                   mydata[obs3[-1],],mydata[obs4[-1],],
                   mydata[obs5[-1],]
                  )


    knn_fit<-knn(train=as.matrix(known[,4]),
                 test=as.matrix(unknown[,4]),
                 cl=known[,1],
                 k=1
                 )
    tab <- table(prediction=knn_fit,truth=unknown[,1])
    acc[i]<-sum(diag(tab))/length(unknown[,1])

  }

  #using only SP
  acc_info[j,1]<-mean(acc)

  #quantile
  acc_info[j,2:3]<-quantile(acc, probs=c(0.025, 0.975))

  mean_acc[j]<-mean(acc)
}

#overall
t.test(mean_acc)

#se
se_acc<-se(mean_acc)

#h
h <- se_acc*qt(0.975, 9)
h
#in percetage
h*100

mean(mean_acc)+h

#shapiro wilk test
shapiro.test(mean_acc)

#Kolmogorov-Smirnov test
ks.test(mean_acc, 'pnorm')

#Anderson-Darling
ad.test(mean_acc)

#other info for each episode
acc_info

#
