#data preprocessing
from keras.utils import to_categorical

#building the model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


#my imported packages
import numpy as np#typical package (Kinser 2018)
import itertools as it #I don't think that this is used
import scipy.misc as sm #typical image package (Kinser 2018)
import convert as cvt #custom functions
import rpy2.robjects as robjects #interface with R
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import SpatialDropout2D

#loading images


dirs = ["data/bird", "none"]
bird = cvt.GetAllImagesCNN(dirs)
dirs = ["data/bone", "none"]
bone = cvt.GetAllImagesCNN(dirs)

dirs = ["data/brick", "none"]
brick = cvt.GetAllImagesCNN(dirs)
dirs = ["data/cam", "none"]
cam = cvt.GetAllImagesCNN(dirs)
dirs = ["data/cup", "none"]
cup = cvt.GetAllImagesCNN(dirs)


#creating labels for each class
bird_labels  = [0]*20
bone_labels  = [1]*20
brick_labels = [2]*20

cam_labels = [3]*20
cup_labels = [4]*20

##################################
## training sample size = 4
##################################

#sample size
n=4
#initialize arrays to hold results
train_vals=np.zeros(100)
test_vals=np.zeros(100)

#splitting data into training and testing
rans = robjects.r("""
set.seed(9318)
n=4
#training datasets
train1<-matrix(nrow=100, ncol=n)
train2<-matrix(nrow=100, ncol=n)
train3<-matrix(nrow=100, ncol=n)
train4<-matrix(nrow=100, ncol=n)
train5<-matrix(nrow=100, ncol=n)

#testing training sets
test1<-matrix(nrow=100, ncol=20-n)
test2<-matrix(nrow=100, ncol=20-n)
test3<-matrix(nrow=100, ncol=20-n)
test4<-matrix(nrow=100, ncol=20-n)
test5<-matrix(nrow=100, ncol=20-n)

for(i in 1:100){
    temp<-0:19

    train1[i,]<-sample(0:19,  n)
    test1[i,]<-temp[-(train1[i,]+1)]

    train2[i,]<-sample(0:19, n)
    test2[i,]<-temp[-(train2[i,]+1)]

    train3[i,]<-sample(0:19, n)
    test3[i,]<-temp[-(train3[i,]+1)]

    train4[i,]<-sample(0:19, n)
    test4[i,]<-temp[-(train4[i,]+1)]

    train5[i,]<-sample(0:19, n)
    test5[i,]<-temp[-(train5[i,]+1)]

}
""")
a1_ran = np.asarray(robjects.r["train1"])
a2_ran = np.asarray(robjects.r["train2"])
a3_ran = np.asarray(robjects.r["train3"])
a4_ran = np.asarray(robjects.r["train4"])
a5_ran = np.asarray(robjects.r["train5"])

a1_test = np.asarray(robjects.r["test1"])
a2_test = np.asarray(robjects.r["test2"])
a3_test = np.asarray(robjects.r["test3"])
a4_test = np.asarray(robjects.r["test4"])
a5_test = np.asarray(robjects.r["test5"])

for j in range(0,100):
    #############
    #all classes
    #############
    #training
    #training y labels
    a1_y_train =  [bird_labels[i] for i in a1_ran[j]]
    a2_y_train =  [bone_labels[i] for i in a2_ran[j]]
    a3_y_train =  [brick_labels[i] for i in a3_ran[j]]
    a4_y_train =  [cam_labels[i] for i in a4_ran[j]]
    a5_y_train =  [cup_labels[i] for i in a5_ran[j]]
    y_train = np.concatenate( (a1_y_train, a2_y_train, a3_y_train, a4_y_train, a5_y_train), axis=0 )
    #training images
    a1_x_train =  [bird[i] for i in a1_ran[j]]
    a2_x_train =  [bone[i] for i in a2_ran[j]]
    a3_x_train =  [brick[i] for i in a3_ran[j]]
    a4_x_train =  [cam[i] for i in a4_ran[j]]
    a5_x_train =  [cup[i] for i in a5_ran[j]]
    X_train = np.concatenate( (a1_x_train, a2_x_train, a3_x_train, a4_x_train, a5_x_train), axis=0 )
    #testing
    #testing y labels
    a1_y_test =  [bird_labels[i] for i in a1_test[j]]
    a2_y_test =  [bone_labels[i] for i in a2_test[j]]
    a3_y_test =  [brick_labels[i] for i in a3_test[j]]
    a4_y_test =  [cam_labels[i] for i in a4_test[j]]
    a5_y_test =  [cup_labels[i] for i in a5_test[j]]
    y_test = np.concatenate( (a1_y_test, a2_y_test, a3_y_test, a4_y_test, a5_y_test), axis=0 )
    #testing images
    a1_x_test =  [bird[i] for i in a1_test[j]]
    a2_x_test =  [bone[i] for i in a2_test[j]]
    a3_x_test =  [brick[i] for i in a3_test[j]]
    a4_x_test =  [cam[i] for i in a4_test[j]]
    a5_x_test =  [cup[i] for i in a5_test[j]]
    X_test = np.concatenate( (a1_x_test, a2_x_test, a3_x_test, a4_x_test, a5_x_test), axis=0 )
    ##################
    #data pre-processing
    ##################
    #reshape data to fit model
    X_train = X_train.reshape(len(X_train),120,120,1)
    X_test = X_test.reshape(len(X_test),120,120,1)
    #convert
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    #y_train[0]
    ##################
    #building the model
    ##################
    #3 conv layers
    #3 pooling layers
    #with epochs with early stopping
    #create model
    model = Sequential()
    #add early stopping component
    ES = [EarlyStopping(monitor='loss', min_delta=0.001, patience=10)]
    #add model layers
    #first C layer
    #Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 120x120x1 images
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(120, 120, 1), activity_regularizer=l2(0.001) ))
    #first pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #first spatial dropout
    model.add(SpatialDropout2D(0.2))
    #second C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001)))
    #second pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #second spatial dropout
    model.add(SpatialDropout2D(0.2))
    #third C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001)))
    #third pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #third spatial dropout
    model.add(SpatialDropout2D(0.2))
    #adding 'flatten' layer
    model.add(Flatten())
    #6 nodes, softmax activation
    model.add(Dense(5, activation='softmax'))
    ##################
    #compiling the model
    ##################
    #lower score indicates better performance
    #also provides accuracy measure
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ##################
    #training the model
    ##################
    #uses validation dat
    #epoch =100  means that the entire dataset is passed forward and backward through NN 100 times
    # see https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
    model.fit(X_train, y_train, epochs=100, callbacks=ES, verbose=0)
    ##################
    #making predictions
    ##################
    #obtaining testing accuracy
    vals=model.predict(X_train)
    vals_pred = np.argmax(vals, axis=1)
    #checking with truth
    vals_truth = np.argmax(y_train, axis=1)
    train_vals[j]=sum((vals_pred==vals_truth)+0.0) / (len(vals_truth)+0.0)
    #predicting on testing
    vals=model.predict(X_test)
    vals_pred = np.argmax(vals, axis=1)
    #checking with truth
    vals_truth = np.argmax(y_test, axis=1)
    test_vals[j]=sum((vals_pred==vals_truth)+0.0) / (len(vals_truth)+0.0)



print('##################################')

print("Results for n=" + str(n))
print("Mean Training")
print(train_vals.mean())
print("SD Training")
print(train_vals.std())

print("Mean Testing")
print(test_vals.mean())
print("SD Training")
print(test_vals.std())

print('##################################')
