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


#creating labels for each class
tri_labels  = [0]*125
squ_labels  = [1]*125
pent_labels = [2]*125
hex_labels  = [3]*125
hept_labels = [4]*125
oct_labels  = [5]*125

#loading images


dirs = ["tris", "none"]
tri = cvt.GetAllImagesCNN(dirs)
dirs = ["squs", "none"]
squs = cvt.GetAllImagesCNN(dirs)
dirs = ["pens", "none"]
pens = cvt.GetAllImagesCNN(dirs)
dirs = ["hexs", "none"]
hexs = cvt.GetAllImagesCNN(dirs)
dirs = ["hept", "none"]
hepts = cvt.GetAllImagesCNN(dirs)
dirs = ["octs", "none"]
octs = cvt.GetAllImagesCNN(dirs)



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
set.seed(695304)
n=3
#training datasets
train3<-matrix(nrow=100, ncol=n)
train4<-matrix(nrow=100, ncol=n)
train5<-matrix(nrow=100, ncol=n)
train6<-matrix(nrow=100, ncol=n)
train7<-matrix(nrow=100, ncol=n)
train8<-matrix(nrow=100, ncol=n)
#testing training sets
test3<-matrix(nrow=100, ncol=125-n)
test4<-matrix(nrow=100, ncol=125-n)
test5<-matrix(nrow=100, ncol=125-n)
test6<-matrix(nrow=100, ncol=125-n)
test7<-matrix(nrow=100, ncol=125-n)
test8<-matrix(nrow=100, ncol=125-n)
for(i in 1:100){
    temp<-0:124
    train3[i,]<-sample(0:124,  n)
    test3[i,]<-temp[-(train3[i,]+1)]
    train4[i,]<-sample(0:124,  n)
    test4[i,]<-temp[-(train4[i,]+1)]
    train5[i,]<-sample(0:124,  n)
    test5[i,]<-temp[-(train5[i,]+1)]
    train6[i,]<-sample(0:124,  n)
    test6[i,]<-temp[-(train6[i,]+1)]
    train7[i,]<-sample(0:124,  n)
    test7[i,]<-temp[-(train7[i,]+1)]
    train8[i,]<-sample(0:124,  n)
    test8[i,]<-temp[-(train8[i,]+1)]
}
""")

tri_ran = np.asarray(robjects.r["train3"])
squ_ran = np.asarray(robjects.r["train4"])
pent_ran = np.asarray(robjects.r["train5"])
hex_ran = np.asarray(robjects.r["train6"])
hept_ran = np.asarray(robjects.r["train7"])
oct_ran = np.asarray(robjects.r["train8"])

tri_test = np.asarray(robjects.r["test3"])
squ_test = np.asarray(robjects.r["test4"])
pent_test = np.asarray(robjects.r["test5"])
hex_test = np.asarray(robjects.r["test6"])
hept_test = np.asarray(robjects.r["test7"])
oct_test = np.asarray(robjects.r["test8"])

for j in range(0,100):
    #training
    #trainging y labels
    tri_y_train =  [tri_labels[i] for i in tri_ran[j]]
    squ_y_train =  [squ_labels[i] for i in squ_ran[j]]
    pent_y_train =  [pent_labels[i] for i in pent_ran[j]]
    hex_y_train =  [hex_labels[i] for i in hex_ran[j]]
    hept_y_train =  [hept_labels[i] for i in hept_ran[j]]
    oct_y_train =  [oct_labels[i] for i in oct_ran[j]]
    y_train = np.concatenate( (tri_y_train, squ_y_train, pent_y_train, hex_y_train, hept_y_train, oct_y_train), axis=0 )
    #x training data
    tri_x_train =  [tri[i] for i in tri_ran[j]]
    squ_x_train =  [squs[i] for i in squ_ran[j]]
    pent_x_train =  [pens[i] for i in pent_ran[j]]
    hex_x_train =  [hexs[i] for i in hex_ran[j]]
    hept_x_train =  [hepts[i] for i in hept_ran[j]]
    oct_x_train =  [octs[i] for i in oct_ran[j]]
    X_train = np.concatenate( (tri_x_train, squ_x_train, pent_x_train, hex_x_train, hept_x_train, oct_x_train), axis=0 )
    #testing
    #testing y labels
    tri_y_test =  [tri_labels[i] for i in tri_test[j]]
    squ_y_test =  [squ_labels[i] for i in squ_test[j]]
    pent_y_test =  [pent_labels[i] for i in pent_test[j]]
    hex_y_test =  [hex_labels[i] for i in hex_test[j]]
    hept_y_test =  [hept_labels[i] for i in hept_test[j]]
    oct_y_test =  [oct_labels[i] for i in oct_test[j]]
    y_test = np.concatenate( (tri_y_test, squ_y_test, pent_y_test, hex_y_test, hept_y_test, oct_y_test), axis=0 )
    #x testing data
    tri_x_test =  [tri[i] for i in tri_test[j]]
    squ_x_test =  [squs[i] for i in squ_test[j]]
    pent_x_test =  [pens[i] for i in pent_test[j]]
    hex_x_test =  [hexs[i] for i in hex_test[j]]
    hept_x_test =  [hepts[i] for i in hept_test[j]]
    oct_x_test =  [octs[i] for i in oct_test[j]]
    X_test = np.concatenate( (tri_x_test, squ_x_test, pent_x_test, hex_x_test, hept_x_test, oct_x_test), axis=0 )
    ##################
    #data pre-processing
    ##################
    #reshape data to fit model
    X_train = X_train.reshape(len(X_train),316,316,1)
    X_test = X_test.reshape(len(X_test),316,316,1)
    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
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
    #Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 316x316x1 images
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(316, 316, 1), activity_regularizer=l2(0.001) ))
    #first pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #first spatial dropout
    model.add(SpatialDropout2D(0.2))
    #second C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #second pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) ))
    #second spatial dropout
    model.add(SpatialDropout2D(0.2))
    #third C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #third pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #third spatial dropout
    model.add(SpatialDropout2D(0.2))
    #adding 'flatten' layer
    model.add(Flatten())
    #6 nodes, softmax activation
    model.add(Dense(6, activation='softmax'))
    ##################
    #compiling the model
    ##################
    #lower score indicates better performance
    #also provides accuracy measure
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ##################
    #training the model
    ##################
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
    #print(j)


#

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
set.seed(555665)
n=4
#training datasets
train3<-matrix(nrow=100, ncol=n)
train4<-matrix(nrow=100, ncol=n)
train5<-matrix(nrow=100, ncol=n)
train6<-matrix(nrow=100, ncol=n)
train7<-matrix(nrow=100, ncol=n)
train8<-matrix(nrow=100, ncol=n)
#testing training sets
test3<-matrix(nrow=100, ncol=125-n)
test4<-matrix(nrow=100, ncol=125-n)
test5<-matrix(nrow=100, ncol=125-n)
test6<-matrix(nrow=100, ncol=125-n)
test7<-matrix(nrow=100, ncol=125-n)
test8<-matrix(nrow=100, ncol=125-n)
for(i in 1:100){
    temp<-0:124
    train3[i,]<-sample(0:124,  n)
    test3[i,]<-temp[-(train3[i,]+1)]
    train4[i,]<-sample(0:124,  n)
    test4[i,]<-temp[-(train4[i,]+1)]
    train5[i,]<-sample(0:124,  n)
    test5[i,]<-temp[-(train5[i,]+1)]
    train6[i,]<-sample(0:124,  n)
    test6[i,]<-temp[-(train6[i,]+1)]
    train7[i,]<-sample(0:124,  n)
    test7[i,]<-temp[-(train7[i,]+1)]
    train8[i,]<-sample(0:124,  n)
    test8[i,]<-temp[-(train8[i,]+1)]
}
""")

tri_ran = np.asarray(robjects.r["train3"])
squ_ran = np.asarray(robjects.r["train4"])
pent_ran = np.asarray(robjects.r["train5"])
hex_ran = np.asarray(robjects.r["train6"])
hept_ran = np.asarray(robjects.r["train7"])
oct_ran = np.asarray(robjects.r["train8"])

tri_test = np.asarray(robjects.r["test3"])
squ_test = np.asarray(robjects.r["test4"])
pent_test = np.asarray(robjects.r["test5"])
hex_test = np.asarray(robjects.r["test6"])
hept_test = np.asarray(robjects.r["test7"])
oct_test = np.asarray(robjects.r["test8"])

for j in range(0,100):
    #training
    #trainging y labels
    tri_y_train =  [tri_labels[i] for i in tri_ran[j]]
    squ_y_train =  [squ_labels[i] for i in squ_ran[j]]
    pent_y_train =  [pent_labels[i] for i in pent_ran[j]]
    hex_y_train =  [hex_labels[i] for i in hex_ran[j]]
    hept_y_train =  [hept_labels[i] for i in hept_ran[j]]
    oct_y_train =  [oct_labels[i] for i in oct_ran[j]]
    y_train = np.concatenate( (tri_y_train, squ_y_train, pent_y_train, hex_y_train, hept_y_train, oct_y_train), axis=0 )
    #x training data
    tri_x_train =  [tri[i] for i in tri_ran[j]]
    squ_x_train =  [squs[i] for i in squ_ran[j]]
    pent_x_train =  [pens[i] for i in pent_ran[j]]
    hex_x_train =  [hexs[i] for i in hex_ran[j]]
    hept_x_train =  [hepts[i] for i in hept_ran[j]]
    oct_x_train =  [octs[i] for i in oct_ran[j]]
    X_train = np.concatenate( (tri_x_train, squ_x_train, pent_x_train, hex_x_train, hept_x_train, oct_x_train), axis=0 )
    #testing
    #testing y labels
    tri_y_test =  [tri_labels[i] for i in tri_test[j]]
    squ_y_test =  [squ_labels[i] for i in squ_test[j]]
    pent_y_test =  [pent_labels[i] for i in pent_test[j]]
    hex_y_test =  [hex_labels[i] for i in hex_test[j]]
    hept_y_test =  [hept_labels[i] for i in hept_test[j]]
    oct_y_test =  [oct_labels[i] for i in oct_test[j]]
    y_test = np.concatenate( (tri_y_test, squ_y_test, pent_y_test, hex_y_test, hept_y_test, oct_y_test), axis=0 )
    #x testing data
    tri_x_test =  [tri[i] for i in tri_test[j]]
    squ_x_test =  [squs[i] for i in squ_test[j]]
    pent_x_test =  [pens[i] for i in pent_test[j]]
    hex_x_test =  [hexs[i] for i in hex_test[j]]
    hept_x_test =  [hepts[i] for i in hept_test[j]]
    oct_x_test =  [octs[i] for i in oct_test[j]]
    X_test = np.concatenate( (tri_x_test, squ_x_test, pent_x_test, hex_x_test, hept_x_test, oct_x_test), axis=0 )
    ##################
    #data pre-processing
    ##################
    #reshape data to fit model
    X_train = X_train.reshape(len(X_train),316,316,1)
    X_test = X_test.reshape(len(X_test),316,316,1)
    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
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
    #Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 316x316x1 images
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(316, 316, 1), activity_regularizer=l2(0.001) ))
    #first pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #first spatial dropout
    model.add(SpatialDropout2D(0.2))
    #second C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #second pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) ))
    #second spatial dropout
    model.add(SpatialDropout2D(0.2))
    #third C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #third pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #third spatial dropout
    model.add(SpatialDropout2D(0.2))
    #adding 'flatten' layer
    model.add(Flatten())
    #6 nodes, softmax activation
    model.add(Dense(6, activation='softmax'))
    ##################
    #compiling the model
    ##################
    #lower score indicates better performance
    #also provides accuracy measure
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ##################
    #training the model
    ##################
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
    #print(j)


#

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
set.seed(723019)
n=5
#training datasets
train3<-matrix(nrow=100, ncol=n)
train4<-matrix(nrow=100, ncol=n)
train5<-matrix(nrow=100, ncol=n)
train6<-matrix(nrow=100, ncol=n)
train7<-matrix(nrow=100, ncol=n)
train8<-matrix(nrow=100, ncol=n)
#testing training sets
test3<-matrix(nrow=100, ncol=125-n)
test4<-matrix(nrow=100, ncol=125-n)
test5<-matrix(nrow=100, ncol=125-n)
test6<-matrix(nrow=100, ncol=125-n)
test7<-matrix(nrow=100, ncol=125-n)
test8<-matrix(nrow=100, ncol=125-n)
for(i in 1:100){
    temp<-0:124
    train3[i,]<-sample(0:124,  n)
    test3[i,]<-temp[-(train3[i,]+1)]
    train4[i,]<-sample(0:124,  n)
    test4[i,]<-temp[-(train4[i,]+1)]
    train5[i,]<-sample(0:124,  n)
    test5[i,]<-temp[-(train5[i,]+1)]
    train6[i,]<-sample(0:124,  n)
    test6[i,]<-temp[-(train6[i,]+1)]
    train7[i,]<-sample(0:124,  n)
    test7[i,]<-temp[-(train7[i,]+1)]
    train8[i,]<-sample(0:124,  n)
    test8[i,]<-temp[-(train8[i,]+1)]
}
""")

tri_ran = np.asarray(robjects.r["train3"])
squ_ran = np.asarray(robjects.r["train4"])
pent_ran = np.asarray(robjects.r["train5"])
hex_ran = np.asarray(robjects.r["train6"])
hept_ran = np.asarray(robjects.r["train7"])
oct_ran = np.asarray(robjects.r["train8"])

tri_test = np.asarray(robjects.r["test3"])
squ_test = np.asarray(robjects.r["test4"])
pent_test = np.asarray(robjects.r["test5"])
hex_test = np.asarray(robjects.r["test6"])
hept_test = np.asarray(robjects.r["test7"])
oct_test = np.asarray(robjects.r["test8"])

for j in range(0,100):
    #training
    #trainging y labels
    tri_y_train =  [tri_labels[i] for i in tri_ran[j]]
    squ_y_train =  [squ_labels[i] for i in squ_ran[j]]
    pent_y_train =  [pent_labels[i] for i in pent_ran[j]]
    hex_y_train =  [hex_labels[i] for i in hex_ran[j]]
    hept_y_train =  [hept_labels[i] for i in hept_ran[j]]
    oct_y_train =  [oct_labels[i] for i in oct_ran[j]]
    y_train = np.concatenate( (tri_y_train, squ_y_train, pent_y_train, hex_y_train, hept_y_train, oct_y_train), axis=0 )
    #x training data
    tri_x_train =  [tri[i] for i in tri_ran[j]]
    squ_x_train =  [squs[i] for i in squ_ran[j]]
    pent_x_train =  [pens[i] for i in pent_ran[j]]
    hex_x_train =  [hexs[i] for i in hex_ran[j]]
    hept_x_train =  [hepts[i] for i in hept_ran[j]]
    oct_x_train =  [octs[i] for i in oct_ran[j]]
    X_train = np.concatenate( (tri_x_train, squ_x_train, pent_x_train, hex_x_train, hept_x_train, oct_x_train), axis=0 )
    #testing
    #testing y labels
    tri_y_test =  [tri_labels[i] for i in tri_test[j]]
    squ_y_test =  [squ_labels[i] for i in squ_test[j]]
    pent_y_test =  [pent_labels[i] for i in pent_test[j]]
    hex_y_test =  [hex_labels[i] for i in hex_test[j]]
    hept_y_test =  [hept_labels[i] for i in hept_test[j]]
    oct_y_test =  [oct_labels[i] for i in oct_test[j]]
    y_test = np.concatenate( (tri_y_test, squ_y_test, pent_y_test, hex_y_test, hept_y_test, oct_y_test), axis=0 )
    #x testing data
    tri_x_test =  [tri[i] for i in tri_test[j]]
    squ_x_test =  [squs[i] for i in squ_test[j]]
    pent_x_test =  [pens[i] for i in pent_test[j]]
    hex_x_test =  [hexs[i] for i in hex_test[j]]
    hept_x_test =  [hepts[i] for i in hept_test[j]]
    oct_x_test =  [octs[i] for i in oct_test[j]]
    X_test = np.concatenate( (tri_x_test, squ_x_test, pent_x_test, hex_x_test, hept_x_test, oct_x_test), axis=0 )
    ##################
    #data pre-processing
    ##################
    #reshape data to fit model
    X_train = X_train.reshape(len(X_train),316,316,1)
    X_test = X_test.reshape(len(X_test),316,316,1)
    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
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
    #Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 316x316x1 images
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(316, 316, 1), activity_regularizer=l2(0.001) ))
    #first pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #first spatial dropout
    model.add(SpatialDropout2D(0.2))
    #second C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #second pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) ))
    #second spatial dropout
    model.add(SpatialDropout2D(0.2))
    #third C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #third pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #third spatial dropout
    model.add(SpatialDropout2D(0.2))
    #adding 'flatten' layer
    model.add(Flatten())
    #6 nodes, softmax activation
    model.add(Dense(6, activation='softmax'))
    ##################
    #compiling the model
    ##################
    #lower score indicates better performance
    #also provides accuracy measure
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ##################
    #training the model
    ##################
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
    #print(j)


#

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
## training sample size = 6
##################################

#sample size
n=6
#initialize arrays to hold results
train_vals=np.zeros(100)
test_vals=np.zeros(100)

#splitting data into training and testing
rans = robjects.r("""
set.seed(442644)
n=6
#training datasets
train3<-matrix(nrow=100, ncol=n)
train4<-matrix(nrow=100, ncol=n)
train5<-matrix(nrow=100, ncol=n)
train6<-matrix(nrow=100, ncol=n)
train7<-matrix(nrow=100, ncol=n)
train8<-matrix(nrow=100, ncol=n)
#testing training sets
test3<-matrix(nrow=100, ncol=125-n)
test4<-matrix(nrow=100, ncol=125-n)
test5<-matrix(nrow=100, ncol=125-n)
test6<-matrix(nrow=100, ncol=125-n)
test7<-matrix(nrow=100, ncol=125-n)
test8<-matrix(nrow=100, ncol=125-n)
for(i in 1:100){
    temp<-0:124
    train3[i,]<-sample(0:124,  n)
    test3[i,]<-temp[-(train3[i,]+1)]
    train4[i,]<-sample(0:124,  n)
    test4[i,]<-temp[-(train4[i,]+1)]
    train5[i,]<-sample(0:124,  n)
    test5[i,]<-temp[-(train5[i,]+1)]
    train6[i,]<-sample(0:124,  n)
    test6[i,]<-temp[-(train6[i,]+1)]
    train7[i,]<-sample(0:124,  n)
    test7[i,]<-temp[-(train7[i,]+1)]
    train8[i,]<-sample(0:124,  n)
    test8[i,]<-temp[-(train8[i,]+1)]
}
""")

tri_ran = np.asarray(robjects.r["train3"])
squ_ran = np.asarray(robjects.r["train4"])
pent_ran = np.asarray(robjects.r["train5"])
hex_ran = np.asarray(robjects.r["train6"])
hept_ran = np.asarray(robjects.r["train7"])
oct_ran = np.asarray(robjects.r["train8"])

tri_test = np.asarray(robjects.r["test3"])
squ_test = np.asarray(robjects.r["test4"])
pent_test = np.asarray(robjects.r["test5"])
hex_test = np.asarray(robjects.r["test6"])
hept_test = np.asarray(robjects.r["test7"])
oct_test = np.asarray(robjects.r["test8"])

for j in range(0,100):
    #training
    #trainging y labels
    tri_y_train =  [tri_labels[i] for i in tri_ran[j]]
    squ_y_train =  [squ_labels[i] for i in squ_ran[j]]
    pent_y_train =  [pent_labels[i] for i in pent_ran[j]]
    hex_y_train =  [hex_labels[i] for i in hex_ran[j]]
    hept_y_train =  [hept_labels[i] for i in hept_ran[j]]
    oct_y_train =  [oct_labels[i] for i in oct_ran[j]]
    y_train = np.concatenate( (tri_y_train, squ_y_train, pent_y_train, hex_y_train, hept_y_train, oct_y_train), axis=0 )
    #x training data
    tri_x_train =  [tri[i] for i in tri_ran[j]]
    squ_x_train =  [squs[i] for i in squ_ran[j]]
    pent_x_train =  [pens[i] for i in pent_ran[j]]
    hex_x_train =  [hexs[i] for i in hex_ran[j]]
    hept_x_train =  [hepts[i] for i in hept_ran[j]]
    oct_x_train =  [octs[i] for i in oct_ran[j]]
    X_train = np.concatenate( (tri_x_train, squ_x_train, pent_x_train, hex_x_train, hept_x_train, oct_x_train), axis=0 )
    #testing
    #testing y labels
    tri_y_test =  [tri_labels[i] for i in tri_test[j]]
    squ_y_test =  [squ_labels[i] for i in squ_test[j]]
    pent_y_test =  [pent_labels[i] for i in pent_test[j]]
    hex_y_test =  [hex_labels[i] for i in hex_test[j]]
    hept_y_test =  [hept_labels[i] for i in hept_test[j]]
    oct_y_test =  [oct_labels[i] for i in oct_test[j]]
    y_test = np.concatenate( (tri_y_test, squ_y_test, pent_y_test, hex_y_test, hept_y_test, oct_y_test), axis=0 )
    #x testing data
    tri_x_test =  [tri[i] for i in tri_test[j]]
    squ_x_test =  [squs[i] for i in squ_test[j]]
    pent_x_test =  [pens[i] for i in pent_test[j]]
    hex_x_test =  [hexs[i] for i in hex_test[j]]
    hept_x_test =  [hepts[i] for i in hept_test[j]]
    oct_x_test =  [octs[i] for i in oct_test[j]]
    X_test = np.concatenate( (tri_x_test, squ_x_test, pent_x_test, hex_x_test, hept_x_test, oct_x_test), axis=0 )
    ##################
    #data pre-processing
    ##################
    #reshape data to fit model
    X_train = X_train.reshape(len(X_train),316,316,1)
    X_test = X_test.reshape(len(X_test),316,316,1)
    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
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
    #Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 316x316x1 images
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(316, 316, 1), activity_regularizer=l2(0.001) ))
    #first pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #first spatial dropout
    model.add(SpatialDropout2D(0.2))
    #second C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #second pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) ))
    #second spatial dropout
    model.add(SpatialDropout2D(0.2))
    #third C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #third pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #third spatial dropout
    model.add(SpatialDropout2D(0.2))
    #adding 'flatten' layer
    model.add(Flatten())
    #6 nodes, softmax activation
    model.add(Dense(6, activation='softmax'))
    ##################
    #compiling the model
    ##################
    #lower score indicates better performance
    #also provides accuracy measure
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ##################
    #training the model
    ##################
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
    #print(j)


#

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
set.seed(459237)
n=7
#training datasets
train3<-matrix(nrow=100, ncol=n)
train4<-matrix(nrow=100, ncol=n)
train5<-matrix(nrow=100, ncol=n)
train6<-matrix(nrow=100, ncol=n)
train7<-matrix(nrow=100, ncol=n)
train8<-matrix(nrow=100, ncol=n)
#testing training sets
test3<-matrix(nrow=100, ncol=125-n)
test4<-matrix(nrow=100, ncol=125-n)
test5<-matrix(nrow=100, ncol=125-n)
test6<-matrix(nrow=100, ncol=125-n)
test7<-matrix(nrow=100, ncol=125-n)
test8<-matrix(nrow=100, ncol=125-n)
for(i in 1:100){
    temp<-0:124
    train3[i,]<-sample(0:124,  n)
    test3[i,]<-temp[-(train3[i,]+1)]
    train4[i,]<-sample(0:124,  n)
    test4[i,]<-temp[-(train4[i,]+1)]
    train5[i,]<-sample(0:124,  n)
    test5[i,]<-temp[-(train5[i,]+1)]
    train6[i,]<-sample(0:124,  n)
    test6[i,]<-temp[-(train6[i,]+1)]
    train7[i,]<-sample(0:124,  n)
    test7[i,]<-temp[-(train7[i,]+1)]
    train8[i,]<-sample(0:124,  n)
    test8[i,]<-temp[-(train8[i,]+1)]
}
""")

tri_ran = np.asarray(robjects.r["train3"])
squ_ran = np.asarray(robjects.r["train4"])
pent_ran = np.asarray(robjects.r["train5"])
hex_ran = np.asarray(robjects.r["train6"])
hept_ran = np.asarray(robjects.r["train7"])
oct_ran = np.asarray(robjects.r["train8"])

tri_test = np.asarray(robjects.r["test3"])
squ_test = np.asarray(robjects.r["test4"])
pent_test = np.asarray(robjects.r["test5"])
hex_test = np.asarray(robjects.r["test6"])
hept_test = np.asarray(robjects.r["test7"])
oct_test = np.asarray(robjects.r["test8"])

for j in range(0,100):
    #training
    #trainging y labels
    tri_y_train =  [tri_labels[i] for i in tri_ran[j]]
    squ_y_train =  [squ_labels[i] for i in squ_ran[j]]
    pent_y_train =  [pent_labels[i] for i in pent_ran[j]]
    hex_y_train =  [hex_labels[i] for i in hex_ran[j]]
    hept_y_train =  [hept_labels[i] for i in hept_ran[j]]
    oct_y_train =  [oct_labels[i] for i in oct_ran[j]]
    y_train = np.concatenate( (tri_y_train, squ_y_train, pent_y_train, hex_y_train, hept_y_train, oct_y_train), axis=0 )
    #x training data
    tri_x_train =  [tri[i] for i in tri_ran[j]]
    squ_x_train =  [squs[i] for i in squ_ran[j]]
    pent_x_train =  [pens[i] for i in pent_ran[j]]
    hex_x_train =  [hexs[i] for i in hex_ran[j]]
    hept_x_train =  [hepts[i] for i in hept_ran[j]]
    oct_x_train =  [octs[i] for i in oct_ran[j]]
    X_train = np.concatenate( (tri_x_train, squ_x_train, pent_x_train, hex_x_train, hept_x_train, oct_x_train), axis=0 )
    #testing
    #testing y labels
    tri_y_test =  [tri_labels[i] for i in tri_test[j]]
    squ_y_test =  [squ_labels[i] for i in squ_test[j]]
    pent_y_test =  [pent_labels[i] for i in pent_test[j]]
    hex_y_test =  [hex_labels[i] for i in hex_test[j]]
    hept_y_test =  [hept_labels[i] for i in hept_test[j]]
    oct_y_test =  [oct_labels[i] for i in oct_test[j]]
    y_test = np.concatenate( (tri_y_test, squ_y_test, pent_y_test, hex_y_test, hept_y_test, oct_y_test), axis=0 )
    #x testing data
    tri_x_test =  [tri[i] for i in tri_test[j]]
    squ_x_test =  [squs[i] for i in squ_test[j]]
    pent_x_test =  [pens[i] for i in pent_test[j]]
    hex_x_test =  [hexs[i] for i in hex_test[j]]
    hept_x_test =  [hepts[i] for i in hept_test[j]]
    oct_x_test =  [octs[i] for i in oct_test[j]]
    X_test = np.concatenate( (tri_x_test, squ_x_test, pent_x_test, hex_x_test, hept_x_test, oct_x_test), axis=0 )
    ##################
    #data pre-processing
    ##################
    #reshape data to fit model
    X_train = X_train.reshape(len(X_train),316,316,1)
    X_test = X_test.reshape(len(X_test),316,316,1)
    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
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
    #Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 316x316x1 images
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(316, 316, 1), activity_regularizer=l2(0.001) ))
    #first pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #first spatial dropout
    model.add(SpatialDropout2D(0.2))
    #second C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #second pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) ))
    #second spatial dropout
    model.add(SpatialDropout2D(0.2))
    #third C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #third pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #third spatial dropout
    model.add(SpatialDropout2D(0.2))
    #adding 'flatten' layer
    model.add(Flatten())
    #6 nodes, softmax activation
    model.add(Dense(6, activation='softmax'))
    ##################
    #compiling the model
    ##################
    #lower score indicates better performance
    #also provides accuracy measure
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ##################
    #training the model
    ##################
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
    #print(j)


#

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
## training sample size = 8
##################################

#sample size
n=8
#initialize arrays to hold results
train_vals=np.zeros(100)
test_vals=np.zeros(100)

#splitting data into training and testing
rans = robjects.r("""
set.seed(326668)
n=8
#training datasets
train3<-matrix(nrow=100, ncol=n)
train4<-matrix(nrow=100, ncol=n)
train5<-matrix(nrow=100, ncol=n)
train6<-matrix(nrow=100, ncol=n)
train7<-matrix(nrow=100, ncol=n)
train8<-matrix(nrow=100, ncol=n)
#testing training sets
test3<-matrix(nrow=100, ncol=125-n)
test4<-matrix(nrow=100, ncol=125-n)
test5<-matrix(nrow=100, ncol=125-n)
test6<-matrix(nrow=100, ncol=125-n)
test7<-matrix(nrow=100, ncol=125-n)
test8<-matrix(nrow=100, ncol=125-n)
for(i in 1:100){
    temp<-0:124
    train3[i,]<-sample(0:124,  n)
    test3[i,]<-temp[-(train3[i,]+1)]
    train4[i,]<-sample(0:124,  n)
    test4[i,]<-temp[-(train4[i,]+1)]
    train5[i,]<-sample(0:124,  n)
    test5[i,]<-temp[-(train5[i,]+1)]
    train6[i,]<-sample(0:124,  n)
    test6[i,]<-temp[-(train6[i,]+1)]
    train7[i,]<-sample(0:124,  n)
    test7[i,]<-temp[-(train7[i,]+1)]
    train8[i,]<-sample(0:124,  n)
    test8[i,]<-temp[-(train8[i,]+1)]
}
""")

tri_ran = np.asarray(robjects.r["train3"])
squ_ran = np.asarray(robjects.r["train4"])
pent_ran = np.asarray(robjects.r["train5"])
hex_ran = np.asarray(robjects.r["train6"])
hept_ran = np.asarray(robjects.r["train7"])
oct_ran = np.asarray(robjects.r["train8"])

tri_test = np.asarray(robjects.r["test3"])
squ_test = np.asarray(robjects.r["test4"])
pent_test = np.asarray(robjects.r["test5"])
hex_test = np.asarray(robjects.r["test6"])
hept_test = np.asarray(robjects.r["test7"])
oct_test = np.asarray(robjects.r["test8"])

for j in range(0,100):
    #training
    #trainging y labels
    tri_y_train =  [tri_labels[i] for i in tri_ran[j]]
    squ_y_train =  [squ_labels[i] for i in squ_ran[j]]
    pent_y_train =  [pent_labels[i] for i in pent_ran[j]]
    hex_y_train =  [hex_labels[i] for i in hex_ran[j]]
    hept_y_train =  [hept_labels[i] for i in hept_ran[j]]
    oct_y_train =  [oct_labels[i] for i in oct_ran[j]]
    y_train = np.concatenate( (tri_y_train, squ_y_train, pent_y_train, hex_y_train, hept_y_train, oct_y_train), axis=0 )
    #x training data
    tri_x_train =  [tri[i] for i in tri_ran[j]]
    squ_x_train =  [squs[i] for i in squ_ran[j]]
    pent_x_train =  [pens[i] for i in pent_ran[j]]
    hex_x_train =  [hexs[i] for i in hex_ran[j]]
    hept_x_train =  [hepts[i] for i in hept_ran[j]]
    oct_x_train =  [octs[i] for i in oct_ran[j]]
    X_train = np.concatenate( (tri_x_train, squ_x_train, pent_x_train, hex_x_train, hept_x_train, oct_x_train), axis=0 )
    #testing
    #testing y labels
    tri_y_test =  [tri_labels[i] for i in tri_test[j]]
    squ_y_test =  [squ_labels[i] for i in squ_test[j]]
    pent_y_test =  [pent_labels[i] for i in pent_test[j]]
    hex_y_test =  [hex_labels[i] for i in hex_test[j]]
    hept_y_test =  [hept_labels[i] for i in hept_test[j]]
    oct_y_test =  [oct_labels[i] for i in oct_test[j]]
    y_test = np.concatenate( (tri_y_test, squ_y_test, pent_y_test, hex_y_test, hept_y_test, oct_y_test), axis=0 )
    #x testing data
    tri_x_test =  [tri[i] for i in tri_test[j]]
    squ_x_test =  [squs[i] for i in squ_test[j]]
    pent_x_test =  [pens[i] for i in pent_test[j]]
    hex_x_test =  [hexs[i] for i in hex_test[j]]
    hept_x_test =  [hepts[i] for i in hept_test[j]]
    oct_x_test =  [octs[i] for i in oct_test[j]]
    X_test = np.concatenate( (tri_x_test, squ_x_test, pent_x_test, hex_x_test, hept_x_test, oct_x_test), axis=0 )
    ##################
    #data pre-processing
    ##################
    #reshape data to fit model
    X_train = X_train.reshape(len(X_train),316,316,1)
    X_test = X_test.reshape(len(X_test),316,316,1)
    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
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
    #Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 316x316x1 images
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(316, 316, 1), activity_regularizer=l2(0.001) ))
    #first pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #first spatial dropout
    model.add(SpatialDropout2D(0.2))
    #second C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #second pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2) ))
    #second spatial dropout
    model.add(SpatialDropout2D(0.2))
    #third C layer
    #Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
    model.add(Conv2D(32, kernel_size=3, activation='relu', activity_regularizer=l2(0.001) ))
    #third pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #third spatial dropout
    model.add(SpatialDropout2D(0.2))
    #adding 'flatten' layer
    model.add(Flatten())
    #6 nodes, softmax activation
    model.add(Dense(6, activation='softmax'))
    ##################
    #compiling the model
    ##################
    #lower score indicates better performance
    #also provides accuracy measure
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    ##################
    #training the model
    ##################
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
    #print(j)


#

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
