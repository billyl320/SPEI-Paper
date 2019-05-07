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


dirs = ["edge_on", "none"]
edge = cvt.GetAllImagesCNN(dirs)
dirs = ["spiral", "none"]
spiral = cvt.GetAllImagesCNN(dirs)
dirs = ["elip", "none"]
elip = cvt.GetAllImagesCNN(dirs)

#creating labels for each class
edge_labels  = [0]*len(edge)
spiral_labels  = [1]*len(spiral)
elip_labels = [2]*len(elip)

##################################
## training sample size = 3
##################################

#sample size
n=3
#initialize arrays to hold results
train_vals=np.zeros(100)
test_vals=np.zeros(100)

#splitting data into training and testing
rans = robjects.r("""
set.seed(76526)
n=3
#training datasets
train1<-matrix(nrow=100, ncol=3)
train2<-matrix(nrow=100, ncol=3)
train3<-matrix(nrow=100, ncol=3)
#testing training sets
test1<-matrix(nrow=100, ncol=75-n)
test2<-matrix(nrow=100, ncol=223-n)
test3<-matrix(nrow=100, ncol=225-n)
for(i in 1:100){
    train1[i,]<-sample(0:74,  3)
    temp<-0:74
    test1[i,]<-temp[-(train1[i,]+1)]
    train2[i,]<-sample(0:222, 3)
    temp<-0:222
    test2[i,]<-temp[-(train2[i,]+1)]
    train3[i,]<-sample(0:224, 3)
    temp<-0:224
    test3[i,]<-temp[-(train3[i,]+1)]
}
""")
edge_ran = np.asarray(robjects.r["train1"])
spiral_ran = np.asarray(robjects.r["train2"])
elip_ran = np.asarray(robjects.r["train3"])

edge_test = np.asarray(robjects.r["test1"])
spiral_test = np.asarray(robjects.r["test2"])
elip_test = np.asarray(robjects.r["test3"])



for j in range(0,100):
    #############
    #all classes
    #############
    #training
    #training y labels
    edge_y_train =  [edge_labels[i] for i in edge_ran[j]]
    spiral_y_train =  [spiral_labels[i] for i in spiral_ran[j]]
    elip_y_train =  [elip_labels[i] for i in elip_ran[j]]
    y_train = np.concatenate( (edge_y_train,spiral_y_train, elip_y_train), axis=0 )
    #training images
    edge_x_train =  [edge[i] for i in edge_ran[j]]
    spiral_x_train =  [spiral[i] for i in spiral_ran[j]]
    elip_x_train =  [elip[i] for i in elip_ran[j]]
    X_train = np.concatenate( (edge_x_train, spiral_x_train, elip_x_train), axis=0 )
    #testing
    #testing y labels
    edge_y_test =  [edge_labels[i] for i in edge_test[j]]
    spiral_y_test =  [spiral_labels[i] for i in spiral_test[j]]
    elip_y_test =  [elip_labels[i] for i in elip_test[j]]
    y_test = np.concatenate( (edge_y_test, spiral_y_test, elip_y_test), axis=0 )
    #testing images
    edge_x_test =  [edge[i] for i in edge_test[j]]
    spiral_x_test =  [spiral[i] for i in spiral_test[j]]
    elip_x_test =  [elip[i] for i in elip_test[j]]
    X_test = np.concatenate( (edge_x_test, spiral_x_test, elip_x_test), axis=0 )
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
    model.add(Dense(3, activation='softmax'))
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
set.seed(872596)
n=4
#training datasets
train1<-matrix(nrow=100, ncol=n)
train2<-matrix(nrow=100, ncol=n)
train3<-matrix(nrow=100, ncol=n)
#testing training sets
test1<-matrix(nrow=100, ncol=75-n)
test2<-matrix(nrow=100, ncol=223-n)
test3<-matrix(nrow=100, ncol=225-n)
for(i in 1:100){
    train1[i,]<-sample(0:74,  n)
    temp<-0:74
    test1[i,]<-temp[-(train1[i,]+1)]
    train2[i,]<-sample(0:222, n)
    temp<-0:222
    test2[i,]<-temp[-(train2[i,]+1)]
    train3[i,]<-sample(0:224, n)
    temp<-0:224
    test3[i,]<-temp[-(train3[i,]+1)]
}
""")
edge_ran = np.asarray(robjects.r["train1"])
spiral_ran = np.asarray(robjects.r["train2"])
elip_ran = np.asarray(robjects.r["train3"])

edge_test = np.asarray(robjects.r["test1"])
spiral_test = np.asarray(robjects.r["test2"])
elip_test = np.asarray(robjects.r["test3"])



for j in range(0,100):
    #############
    #all classes
    #############
    #training
    #training y labels
    edge_y_train =  [edge_labels[i] for i in edge_ran[j]]
    spiral_y_train =  [spiral_labels[i] for i in spiral_ran[j]]
    elip_y_train =  [elip_labels[i] for i in elip_ran[j]]
    y_train = np.concatenate( (edge_y_train,spiral_y_train, elip_y_train), axis=0 )
    #training images
    edge_x_train =  [edge[i] for i in edge_ran[j]]
    spiral_x_train =  [spiral[i] for i in spiral_ran[j]]
    elip_x_train =  [elip[i] for i in elip_ran[j]]
    X_train = np.concatenate( (edge_x_train, spiral_x_train, elip_x_train), axis=0 )
    #testing
    #testing y labels
    edge_y_test =  [edge_labels[i] for i in edge_test[j]]
    spiral_y_test =  [spiral_labels[i] for i in spiral_test[j]]
    elip_y_test =  [elip_labels[i] for i in elip_test[j]]
    y_test = np.concatenate( (edge_y_test, spiral_y_test, elip_y_test), axis=0 )
    #testing images
    edge_x_test =  [edge[i] for i in edge_test[j]]
    spiral_x_test =  [spiral[i] for i in spiral_test[j]]
    elip_x_test =  [elip[i] for i in elip_test[j]]
    X_test = np.concatenate( (edge_x_test, spiral_x_test, elip_x_test), axis=0 )
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
    model.add(Dense(3, activation='softmax'))
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

##################################
## training sample size = 5
##################################

#sample size
n=5
#initialize arrays to hold results
train_vals=np.zeros(100)
test_vals=np.zeros(100)

#splitting data into training and testing
rans = robjects.r("""
set.seed(50976)
n=5
#training datasets
train1<-matrix(nrow=100, ncol=n)
train2<-matrix(nrow=100, ncol=n)
train3<-matrix(nrow=100, ncol=n)
#testing training sets
test1<-matrix(nrow=100, ncol=75-n)
test2<-matrix(nrow=100, ncol=223-n)
test3<-matrix(nrow=100, ncol=225-n)
for(i in 1:100){
    train1[i,]<-sample(0:74,  n)
    temp<-0:74
    test1[i,]<-temp[-(train1[i,]+1)]
    train2[i,]<-sample(0:222, n)
    temp<-0:222
    test2[i,]<-temp[-(train2[i,]+1)]
    train3[i,]<-sample(0:224, n)
    temp<-0:224
    test3[i,]<-temp[-(train3[i,]+1)]
}
""")
edge_ran = np.asarray(robjects.r["train1"])
spiral_ran = np.asarray(robjects.r["train2"])
elip_ran = np.asarray(robjects.r["train3"])

edge_test = np.asarray(robjects.r["test1"])
spiral_test = np.asarray(robjects.r["test2"])
elip_test = np.asarray(robjects.r["test3"])



for j in range(0,100):
    #############
    #all classes
    #############
    #training
    #training y labels
    edge_y_train =  [edge_labels[i] for i in edge_ran[j]]
    spiral_y_train =  [spiral_labels[i] for i in spiral_ran[j]]
    elip_y_train =  [elip_labels[i] for i in elip_ran[j]]
    y_train = np.concatenate( (edge_y_train,spiral_y_train, elip_y_train), axis=0 )
    #training images
    edge_x_train =  [edge[i] for i in edge_ran[j]]
    spiral_x_train =  [spiral[i] for i in spiral_ran[j]]
    elip_x_train =  [elip[i] for i in elip_ran[j]]
    X_train = np.concatenate( (edge_x_train, spiral_x_train, elip_x_train), axis=0 )
    #testing
    #testing y labels
    edge_y_test =  [edge_labels[i] for i in edge_test[j]]
    spiral_y_test =  [spiral_labels[i] for i in spiral_test[j]]
    elip_y_test =  [elip_labels[i] for i in elip_test[j]]
    y_test = np.concatenate( (edge_y_test, spiral_y_test, elip_y_test), axis=0 )
    #testing images
    edge_x_test =  [edge[i] for i in edge_test[j]]
    spiral_x_test =  [spiral[i] for i in spiral_test[j]]
    elip_x_test =  [elip[i] for i in elip_test[j]]
    X_test = np.concatenate( (edge_x_test, spiral_x_test, elip_x_test), axis=0 )
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
    model.add(Dense(3, activation='softmax'))
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

##################################
## training sample size = 7
##################################

#sample size
n=7
#initialize arrays to hold results
train_vals=np.zeros(100)
test_vals=np.zeros(100)

#splitting data into training and testing
rans = robjects.r("""
set.seed(35522)
n=7
#training datasets
train1<-matrix(nrow=100, ncol=n)
train2<-matrix(nrow=100, ncol=n)
train3<-matrix(nrow=100, ncol=n)
#testing training sets
test1<-matrix(nrow=100, ncol=75-n)
test2<-matrix(nrow=100, ncol=223-n)
test3<-matrix(nrow=100, ncol=225-n)
for(i in 1:100){
    train1[i,]<-sample(0:74,  n)
    temp<-0:74
    test1[i,]<-temp[-(train1[i,]+1)]
    train2[i,]<-sample(0:222, n)
    temp<-0:222
    test2[i,]<-temp[-(train2[i,]+1)]
    train3[i,]<-sample(0:224, n)
    temp<-0:224
    test3[i,]<-temp[-(train3[i,]+1)]
}
""")
edge_ran = np.asarray(robjects.r["train1"])
spiral_ran = np.asarray(robjects.r["train2"])
elip_ran = np.asarray(robjects.r["train3"])

edge_test = np.asarray(robjects.r["test1"])
spiral_test = np.asarray(robjects.r["test2"])
elip_test = np.asarray(robjects.r["test3"])



for j in range(0,100):
    #############
    #all classes
    #############
    #training
    #training y labels
    edge_y_train =  [edge_labels[i] for i in edge_ran[j]]
    spiral_y_train =  [spiral_labels[i] for i in spiral_ran[j]]
    elip_y_train =  [elip_labels[i] for i in elip_ran[j]]
    y_train = np.concatenate( (edge_y_train,spiral_y_train, elip_y_train), axis=0 )
    #training images
    edge_x_train =  [edge[i] for i in edge_ran[j]]
    spiral_x_train =  [spiral[i] for i in spiral_ran[j]]
    elip_x_train =  [elip[i] for i in elip_ran[j]]
    X_train = np.concatenate( (edge_x_train, spiral_x_train, elip_x_train), axis=0 )
    #testing
    #testing y labels
    edge_y_test =  [edge_labels[i] for i in edge_test[j]]
    spiral_y_test =  [spiral_labels[i] for i in spiral_test[j]]
    elip_y_test =  [elip_labels[i] for i in elip_test[j]]
    y_test = np.concatenate( (edge_y_test, spiral_y_test, elip_y_test), axis=0 )
    #testing images
    edge_x_test =  [edge[i] for i in edge_test[j]]
    spiral_x_test =  [spiral[i] for i in spiral_test[j]]
    elip_x_test =  [elip[i] for i in elip_test[j]]
    X_test = np.concatenate( (edge_x_test, spiral_x_test, elip_x_test), axis=0 )
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
    model.add(Dense(3, activation='softmax'))
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

##################################
## training sample size = 10
##################################

#sample size
n=10
#initialize arrays to hold results
train_vals=np.zeros(100)
test_vals=np.zeros(100)

#splitting data into training and testing
rans = robjects.r("""
set.seed(1275148)
n=10
#training datasets
train1<-matrix(nrow=100, ncol=n)
train2<-matrix(nrow=100, ncol=n)
train3<-matrix(nrow=100, ncol=n)
#testing training sets
test1<-matrix(nrow=100, ncol=75-n)
test2<-matrix(nrow=100, ncol=223-n)
test3<-matrix(nrow=100, ncol=225-n)
for(i in 1:100){
    train1[i,]<-sample(0:74,  n)
    temp<-0:74
    test1[i,]<-temp[-(train1[i,]+1)]
    train2[i,]<-sample(0:222, n)
    temp<-0:222
    test2[i,]<-temp[-(train2[i,]+1)]
    train3[i,]<-sample(0:224, n)
    temp<-0:224
    test3[i,]<-temp[-(train3[i,]+1)]
}
""")
edge_ran = np.asarray(robjects.r["train1"])
spiral_ran = np.asarray(robjects.r["train2"])
elip_ran = np.asarray(robjects.r["train3"])

edge_test = np.asarray(robjects.r["test1"])
spiral_test = np.asarray(robjects.r["test2"])
elip_test = np.asarray(robjects.r["test3"])



for j in range(0,100):
    #############
    #all classes
    #############
    #training
    #training y labels
    edge_y_train =  [edge_labels[i] for i in edge_ran[j]]
    spiral_y_train =  [spiral_labels[i] for i in spiral_ran[j]]
    elip_y_train =  [elip_labels[i] for i in elip_ran[j]]
    y_train = np.concatenate( (edge_y_train,spiral_y_train, elip_y_train), axis=0 )
    #training images
    edge_x_train =  [edge[i] for i in edge_ran[j]]
    spiral_x_train =  [spiral[i] for i in spiral_ran[j]]
    elip_x_train =  [elip[i] for i in elip_ran[j]]
    X_train = np.concatenate( (edge_x_train, spiral_x_train, elip_x_train), axis=0 )
    #testing
    #testing y labels
    edge_y_test =  [edge_labels[i] for i in edge_test[j]]
    spiral_y_test =  [spiral_labels[i] for i in spiral_test[j]]
    elip_y_test =  [elip_labels[i] for i in elip_test[j]]
    y_test = np.concatenate( (edge_y_test, spiral_y_test, elip_y_test), axis=0 )
    #testing images
    edge_x_test =  [edge[i] for i in edge_test[j]]
    spiral_x_test =  [spiral[i] for i in spiral_test[j]]
    elip_x_test =  [elip[i] for i in elip_test[j]]
    X_test = np.concatenate( (edge_x_test, spiral_x_test, elip_x_test), axis=0 )
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
    model.add(Dense(3, activation='softmax'))
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

##################################
## training sample size = 20
##################################

#sample size
n=20
#initialize arrays to hold results
train_vals=np.zeros(100)
test_vals=np.zeros(100)

#splitting data into training and testing
rans = robjects.r("""
set.seed(5924544)
n=20
#training datasets
train1<-matrix(nrow=100, ncol=n)
train2<-matrix(nrow=100, ncol=n)
train3<-matrix(nrow=100, ncol=n)
#testing training sets
test1<-matrix(nrow=100, ncol=75-n)
test2<-matrix(nrow=100, ncol=223-n)
test3<-matrix(nrow=100, ncol=225-n)
for(i in 1:100){
    train1[i,]<-sample(0:74,  n)
    temp<-0:74
    test1[i,]<-temp[-(train1[i,]+1)]
    train2[i,]<-sample(0:222, n)
    temp<-0:222
    test2[i,]<-temp[-(train2[i,]+1)]
    train3[i,]<-sample(0:224, n)
    temp<-0:224
    test3[i,]<-temp[-(train3[i,]+1)]
}
""")
edge_ran = np.asarray(robjects.r["train1"])
spiral_ran = np.asarray(robjects.r["train2"])
elip_ran = np.asarray(robjects.r["train3"])

edge_test = np.asarray(robjects.r["test1"])
spiral_test = np.asarray(robjects.r["test2"])
elip_test = np.asarray(robjects.r["test3"])



for j in range(0,100):
    #############
    #all classes
    #############
    #training
    #training y labels
    edge_y_train =  [edge_labels[i] for i in edge_ran[j]]
    spiral_y_train =  [spiral_labels[i] for i in spiral_ran[j]]
    elip_y_train =  [elip_labels[i] for i in elip_ran[j]]
    y_train = np.concatenate( (edge_y_train,spiral_y_train, elip_y_train), axis=0 )
    #training images
    edge_x_train =  [edge[i] for i in edge_ran[j]]
    spiral_x_train =  [spiral[i] for i in spiral_ran[j]]
    elip_x_train =  [elip[i] for i in elip_ran[j]]
    X_train = np.concatenate( (edge_x_train, spiral_x_train, elip_x_train), axis=0 )
    #testing
    #testing y labels
    edge_y_test =  [edge_labels[i] for i in edge_test[j]]
    spiral_y_test =  [spiral_labels[i] for i in spiral_test[j]]
    elip_y_test =  [elip_labels[i] for i in elip_test[j]]
    y_test = np.concatenate( (edge_y_test, spiral_y_test, elip_y_test), axis=0 )
    #testing images
    edge_x_test =  [edge[i] for i in edge_test[j]]
    spiral_x_test =  [spiral[i] for i in spiral_test[j]]
    elip_x_test =  [elip[i] for i in elip_test[j]]
    X_test = np.concatenate( (edge_x_test, spiral_x_test, elip_x_test), axis=0 )
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
    model.add(Dense(3, activation='softmax'))
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


#
