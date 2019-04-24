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


dirs = ["triangle", "none"]
tri = cvt.GetAllImagesCNN(dirs)
dirs = ["square", "none"]
squs = cvt.GetAllImagesCNN(dirs)
dirs = ["pent", "reg_pent"]
pens = cvt.GetAllImagesCNN(dirs)
dirs = ["hex","reg_hex"]
hexs = cvt.GetAllImagesCNN(dirs)

dirs = ["circle1", "circle2", "circle3"]
circ = cvt.GetAllImagesCNN(dirs)


#creating labels for each class
tri_labels  = [0]*len(tri)
squ_labels  = [1]*len(squs)
pent_labels = [2]*len(pens)
hex_labels  = [3]*len(hexs)
circ_labels = [4]*len(circ)


#splitting data into training and testing
rans = robjects.r("""
set.seed(83150)

train3<-sample(0:11, 12)
train4<-sample(0:7, 8)
train5<-sample(0:11, 12)
train6<-sample(0:7, 8)
train7<-sample(0:905, 906)

""")

tri_ran = list(robjects.r["train3"])
squ_ran = list(robjects.r["train4"])
pent_ran = list(robjects.r["train5"])
hex_ran = list(robjects.r["train6"])
circ_ran = list(robjects.r["train7"])




#############
#all classes
#############

#training

#trainging y labels
tri_y_train =  [tri_labels[i] for i in tri_ran[0:6]]
squ_y_train =  [squ_labels[i] for i in squ_ran[0:4]]
pent_y_train =  [pent_labels[i] for i in pent_ran[0:6]]
hex_y_train =  [hex_labels[i] for i in hex_ran[0:4]]
circ_y_train =  [circ_labels[i] for i in circ_ran[0:453]]

y_train = np.concatenate( (tri_y_train, squ_y_train, pent_y_train,
                           hex_y_train, circ_y_train), axis=0 )

tri_x_train =  [tri[i] for i in tri_ran[0:6]]
squ_x_train =  [squs[i] for i in squ_ran[0:4]]
pent_x_train =  [pens[i] for i in pent_ran[0:6]]
hex_x_train =  [hexs[i] for i in hex_ran[0:4]]
circ_x_train =  [circ[i] for i in circ_ran[0:453]]

X_train = np.concatenate( (tri_x_train, squ_x_train, pent_x_train,
                           hex_x_train, circ_x_train), axis=0 )



#testing

#testing y labels
tri_y_test =  [tri_labels[i] for i in tri_ran[6:]]
squ_y_test =  [squ_labels[i] for i in squ_ran[4:]]
pent_y_test =  [pent_labels[i] for i in pent_ran[6:]]
hex_y_test =  [hex_labels[i] for i in hex_ran[4:]]
circ_y_test =  [circ_labels[i] for i in circ_ran[453:]]

y_test = np.concatenate( (tri_y_test, squ_y_test, pent_y_test,
                           hex_y_test, circ_y_test), axis=0 )

tri_x_test =  [tri[i] for i in tri_ran[6:]]
squ_x_test =  [squs[i] for i in squ_ran[4:]]
pent_x_test =  [pens[i] for i in pent_ran[6:]]
hex_x_test =  [hexs[i] for i in hex_ran[4:]]
circ_x_test =  [circ[i] for i in circ_ran[453:]]

X_test = np.concatenate( (tri_x_test, squ_x_test, pent_x_test,
                           hex_x_test, circ_x_test), axis=0 )



##################
#data pre-processing
##################

#reshape data to fit model
X_train = X_train.reshape(len(X_train),160,240,1)
X_test = X_test.reshape(len(X_test),160,240,1)

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
#Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 160x240x1 images
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(160, 240, 1)))
#second layer
#Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
model.add(Conv2D(32, kernel_size=3, activation='relu'))

#adding 'flatten' layer
model.add(Flatten())
#5 nodes, softmax activation
model.add(Dense(5, activation='softmax'))


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
#473/473 [==============================] - 55s 115ms/step - loss: 0.7214 - acc: 0.9154
#Epoch 2/3
#473/473 [==============================] - 55s 117ms/step - loss: 0.6815 - acc: 0.9577
#Epoch 3/3
#473/473 [==============================] - 55s 116ms/step - loss: 0.6815 - acc: 0.9577
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
#0.9577167019027484 = 96%


#################
# without circles
#################


#############
#classes
#############

#training

#trainging y labels
y_train = np.concatenate( (tri_y_train, squ_y_train, pent_y_train,
                           hex_y_train), axis=0 )

X_train = np.concatenate( (tri_x_train, squ_x_train, pent_x_train,
                           hex_x_train), axis=0 )



#testing

#testing y labels
y_test = np.concatenate( (tri_y_test, squ_y_test, pent_y_test,
                           hex_y_test), axis=0 )

X_test = np.concatenate( (tri_x_test, squ_x_test, pent_x_test,
                           hex_x_test), axis=0 )



##################
#data pre-processing
##################

#reshape data to fit model
X_train = X_train.reshape(len(X_train),160,240,1)
X_test = X_test.reshape(len(X_test),160,240,1)

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
#Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 160x240x1 images
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(160, 240, 1)))
#second layer
#Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
model.add(Conv2D(32, kernel_size=3, activation='relu'))

#adding 'flatten' layer
model.add(Flatten())
# nodes, softmax activation
model.add(Dense(4, activation='softmax'))


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
#20/20 [==============================] - 3s 143ms/step - loss: 1.3814 - acc: 0.4500
#Epoch 2/3
#20/20 [==============================] - 2s 118ms/step - loss: 2.9443 - acc: 0.6000
#Epoch 3/3
#20/20 [==============================] - 2s 116ms/step - loss: 0.9557 - acc: 0.4000
#<keras.callbacks.History object at 0x11ccc87b8>

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
# 0.30 = 30%

###
#running the model fit but with larger number of epochs
###

#create model
model = Sequential()

#add model layers
#first layer
#Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 160x240x1 images
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(160, 240, 1)))
#second layer
#Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
model.add(Conv2D(32, kernel_size=3, activation='relu'))

#adding 'flatten' layer
model.add(Flatten())
# nodes, softmax activation
model.add(Dense(4, activation='softmax'))

#lower score indicates better performance
#also provides accuracy measure
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])




#uses validation dat
#epoch =10 means that the entire dataset is passed forward and backward through NN 10 times
# see https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
model.fit(X_train, y_train, epochs=10)

#results
#Epoch 1/10
#20/20 [==============================] - 3s 143ms/step - loss: 1.3852 - acc: 0.3000
#Epoch 2/10
#20/20 [==============================] - 2s 115ms/step - loss: 2.1195 - acc: 0.6000
#Epoch 3/10
#20/20 [==============================] - 2s 118ms/step - loss: 0.6504 - acc: 0.9500
#Epoch 4/10
#20/20 [==============================] - 2s 116ms/step - loss: 0.9831 - acc: 0.2500
#Epoch 5/10
#20/20 [==============================] - 2s 115ms/step - loss: 0.3209 - acc: 1.0000
#Epoch 6/10
#20/20 [==============================] - 2s 113ms/step - loss: 0.3282 - acc: 0.9000
#Epoch 7/10
#20/20 [==============================] - 2s 113ms/step - loss: 0.1832 - acc: 1.0000
#Epoch 8/10
#20/20 [==============================] - 2s 114ms/step - loss: 0.1324 - acc: 1.0000
#Epoch 9/10
#20/20 [==============================] - 2s 112ms/step - loss: 0.0802 - acc: 1.0000
#Epoch 10/10
#20/20 [==============================] - 2s 114ms/step - loss: 0.0465 - acc: 1.0000
#<keras.callbacks.History object at 0x11c6c3b00>


##################
#making predictions
##################

#predicting 1st four images
vals=model.predict(X_test)
vals_pred = np.argmax(vals, axis=1)

#checking with truth
vals_truth = np.argmax(y_test, axis=1)

sum((vals_pred==vals_truth)+0.0) / (len(vals_truth)+0.0)
#results
#0.80


###
#running the model fit but with even LARGER number of epochs
###

#create model
model = Sequential()

#add model layers
#first layer
#Rectified Linear Unit activation, 64 nodes, 3x3 filter matrix, 160x240x1 images
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(160, 240, 1)))
#second layer
#Rectified Linear Unit activation, 32 nodes, 3x3 filter matrix
model.add(Conv2D(32, kernel_size=3, activation='relu'))

#adding 'flatten' layer
model.add(Flatten())
# nodes, softmax activation
model.add(Dense(4, activation='softmax'))

#lower score indicates better performance
#also provides accuracy measure
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])




#uses validation dat
#epoch =10 means that the entire dataset is passed forward and backward through NN 10 times
# see https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
model.fit(X_train, y_train, epochs=200)

#results
#basically converged
#Epoch 200/200
#20/20 [==============================] - 2s 110ms/step - loss: 2.8491e-06 - acc: 1.0000

##################
#making predictions
##################

#predicting 1st four images
vals=model.predict(X_test)
vals_pred = np.argmax(vals, axis=1)

#checking with truth
vals_truth = np.argmax(y_test, axis=1)

sum((vals_pred==vals_truth)+0.0) / (len(vals_truth)+0.0)
#results
#0.90


#
