#
#data preprocessing
from keras.utils import to_categorical

#building the model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


#my imported packages
import numpy as np#typical package (Kinser 2018)
import itertools as it #I don't think that this is used
import scipy.misc as sm #typical image package (Kinser 2018)
import convert as cvt #custom functions
import rpy2.robjects as robjects #interface with R


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


#splitting data into training and testing
rans = robjects.r("""
set.seed(50976)

train1<-sample(0:74,  75)
train2<-sample(0:222, 223)
train3<-sample(0:224, 225)


""")

edge_ran = list(robjects.r["train1"])
spiral_ran = list(robjects.r["train2"])
elip_ran = list(robjects.r["train3"])

#############
#all classes
#############

#training

#trainging y labels
edge_y_train =  [edge_labels[i] for i in edge_ran[0:37]]
spiral_y_train =  [spiral_labels[i] for i in spiral_ran[0:111]]
elip_y_train =  [elip_labels[i] for i in elip_ran[0:112]]


y_train = np.concatenate( (edge_y_train,
                           spiral_y_train, elip_y_train), axis=0 )

edge_x_train =  [edge[i] for i in edge_ran[0:37]]
spiral_x_train =  [spiral[i] for i in spiral_ran[0:111]]
elip_x_train =  [elip[i] for i in elip_ran[0:112]]

X_train = np.concatenate( (edge_x_train,
                           spiral_x_train, elip_x_train), axis=0 )



#testing

#testing y labels
edge_y_test =  [edge_labels[i] for i in edge_ran[37:]]
spiral_y_test =  [spiral_labels[i] for i in spiral_ran[111:]]
elip_y_test =  [elip_labels[i] for i in elip_ran[112:]]

y_test = np.concatenate( (edge_y_test,
                          spiral_y_test, elip_y_test), axis=0 )

edge_x_test =  [edge[i] for i in edge_ran[37:]]
spiral_x_test =  [spiral[i] for i in spiral_ran[111:]]
elip_x_test =  [elip[i] for i in elip_ran[112:]]

X_test = np.concatenate( (edge_x_test,
                          spiral_x_test, elip_x_test), axis=0 )



##################
#data pre-processing
##################

#reshape data to fit model
X_train = X_train.reshape(len(X_train),120,120,1)
X_test = X_test.reshape(len(X_test),120,120,1)

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train[0]

##################
#building the model
##################

#create model
model = Sequential()

#add model layers
#first layer
#Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 120x120x1 images
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(120, 120, 1)))
#second layer
#Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
model.add(Conv2D(32, kernel_size=3, activation='relu'))

#adding 'flatten' layer
model.add(Flatten())
#5 nodes, softmax activation
model.add(Dense(3, activation='softmax'))


##################
#compiling the model
##################

#lower score indicates better performance
#also provides accuracy measure
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])



##################
#training the model
##################

#uses validation dat
#epoch =3 means that the entire dataset is passed forward and backward through NN 3 times
# see https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
model.fit(X_train, y_train, epochs=3)

#results
#Epoch 1/3
#260/260 [==============================] - 24s 94ms/step - loss: 8.4424 - acc: 0.3923
#Epoch 2/3
#260/260 [==============================] - 22s 84ms/step - loss: 9.2369 - acc: 0.4269
#Epoch 3/3
#260/260 [==============================] - 22s 83ms/step - loss: 9.2369 - acc: 0.4269
#<keras.callbacks.History object at 0x12115b748>
##################
#making predictions
##################

#predicting 1st four images
vals=model.predict(X_test)
vals_pred = np.argmax(vals, axis=1)

#checking with truth
vals_truth = np.argmax(y_test, axis=1)

sum((vals_pred==vals_truth)+0.0) / (len(vals_truth)+0.0)
#result
#0.42585551330798477 = 43%



###
#running the model fit but with larger number of epochs
###


##################
#building the model
##################

#create model
model = Sequential()

#add model layers
#first layer
#Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 120x120x1 images
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(120, 120, 1)))
#second layer
#Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
model.add(Conv2D(32, kernel_size=3, activation='relu'))

#adding 'flatten' layer
model.add(Flatten())
#5 nodes, softmax activation
model.add(Dense(3, activation='softmax'))


##################
#compiling the model
##################

#lower score indicates better performance
#also provides accuracy measure
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])



##################
#training the model
##################

#uses validation dat
#epoch = 10 means that the entire dataset is passed forward and backward through NN 10 times
# see https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
model.fit(X_train, y_train, epochs=10)

#results
#Epoch 1/10
#260/260 [==============================] - 21s 82ms/step - loss: 8.2640 - acc: 0.3808
#Epoch 2/10
#260/260 [==============================] - 21s 82ms/step - loss: 9.2369 - acc: 0.4269
#Epoch 3/10
#260/260 [==============================] - 21s 82ms/step - loss: 9.2369 - acc: 0.4269
#Epoch 4/10
#260/260 [==============================] - 21s 80ms/step - loss: 9.2369 - acc: 0.4269
#Epoch 5/10
#260/260 [==============================] - 21s 80ms/step - loss: 9.2369 - acc: 0.4269
#Epoch 6/10
#260/260 [==============================] - 21s 80ms/step - loss: 9.2369 - acc: 0.4269
#Epoch 7/10
#260/260 [==============================] - 21s 81ms/step - loss: 9.2369 - acc: 0.4269
#Epoch 8/10
#260/260 [==============================] - 20s 79ms/step - loss: 9.2369 - acc: 0.4269
#Epoch 9/10
#260/260 [==============================] - 21s 81ms/step - loss: 9.2369 - acc: 0.4269
#Epoch 10/10
#260/260 [==============================] - 21s 79ms/step - loss: 9.2369 - acc: 0.4269
#<keras.callbacks.History object at 0x12a3aceb8>

##################
#making predictions
##################

#predicting 1st four images
vals=model.predict(X_test)
vals_pred = np.argmax(vals, axis=1)

#checking with truth
vals_truth = np.argmax(y_test, axis=1)

sum((vals_pred==vals_truth)+0.0) / (len(vals_truth)+0.0)
#result
# 0.42585551330798477 = 43%



###
#seems to have converged - no need to do more epochs
###






#
