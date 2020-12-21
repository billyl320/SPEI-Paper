#outperformance calculations

#####
#poly
#####
spei<-c(0.69, 0.77, 0.80, 0.82, 0.83, 0.83)

#cnn
cnn<-c(0.27, 0.30, 0.32, 0.37, 0.47, 0.46)

#1shot
spei_1<-c(0.902)
cnn_1<-c(0.705)

#####
#pill
#####

spei_pill<-c(0.70, 0.75, 0.80, 0.85)

#cnn
cnn_pill<-c(0.73, 0.72, 0.74, 0.74)

#####
#galy
#####

spei_galy<-c(0.61, 0.62, 0.62, 0.65, 0.67, 0.68)

cnn_galy<-c(0.53, 0.54, 0.55, 0.56, 0.57, 0.60)

######
#mpeg7
######

spei_7<-c(0.81)

cnn_7<-c(0.83)

######
#cell
######

spei_cell<-c(0.981, 0.838, 0.993)

cnn_cell<-c(0.96, 0.869, 0.964)

#######
#sar
#######

spei_sar<-c(0.9571)

cnn_sar<-c(0.91)

#######
#omni
#######

spei_omni<-c(0.317)

cnn_omni<-c(0.996)

##########
#overall
##########

mean(c(spei/cnn, spei_1/cnn_1, spei_pill/cnn_pill,
    spei_galy/cnn_galy, spei_7/cnn_7,
    spei_cell/cnn_cell, spei_sar/cnn_sar,
    spei_omni/cnn_omni
    ))


#
