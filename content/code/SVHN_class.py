import time
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

if os.path.isfile('train_32x32.mat'):
    print('Files exist. Skip downloading.')
else:
    os.system("wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat")
    os.system("wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat")

train = loadmat('train_32x32.mat')
test = loadmat('test_32x32.mat')

x_train = train['X']/255.
y_train = train['y']
x_test = test['X']/255.
y_test = test['y']
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

x_train_gs = np.mean(x_train, axis = 2, keepdims=True)
x_train_gs = x_train_gs.transpose(3,0,1,2)
x_test_gs = np.mean(x_test, axis = 2, keepdims=True)
x_test_gs = x_test_gs.transpose(3,0,1,2)

def get_model(input_shape, dropout_rate, nn_units):
    model = Sequential([Flatten(input_shape = input_shape),
                        Dense(nn_units, activation = 'relu', kernel_initializer = 'he_uniform'),
                        Dropout(dropout_rate),
                        Dense(nn_units, activation = 'relu', kernel_initializer = 'he_uniform'),
                        Dropout(dropout_rate),
                        Dense(nn_units, activation = 'relu', kernel_initializer = 'he_uniform'),
                        Dropout(dropout_rate),
                        Dense(nn_units, activation = 'relu', kernel_initializer = 'he_uniform'),
                        Dropout(dropout_rate),
                        Dense(10, activation = 'softmax')])
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=["accuracy"])
    
    return model


model_mlp = get_model(x_train_gs.shape[1:4],0,64)
earlystop = EarlyStopping(monitor='loss', patience = 5)

epochs = 3
batch_size = 2*64

start = time.time()
history_mlp = model_mlp.fit(x_train_gs, y_train, epochs = epochs, batch_size=batch_size, 
                        validation_split = 0.15, callbacks=[earlystop])
endt = time.time()-start
print("Time for {} epochs: {:0.2f}ms".format(epochs,1000*endt))

def get_model_cnn(input_shape, dropout_rate, nn_units):
    model = Sequential([
        Conv2D(8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform', input_shape = input_shape),
        MaxPooling2D((3,3)),
        Conv2D(16, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform'),
        MaxPooling2D((3,3)),
        Flatten(),
        Dense(nn_units, activation='relu', kernel_initializer = 'he_uniform'),
        Dropout(dropout_rate),
        Dense(nn_units, activation='relu', kernel_initializer = 'he_uniform'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax'),
    ])
    
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    return model

model_cnn = get_model_cnn(x_train_gs.shape[1:4], 0.5, 64)
earlystop = EarlyStopping(monitor='loss', patience = 5)

start = time.time()
history_cnn = model_cnn.fit(x_train_gs, y_train, epochs = epochs, batch_size=batch_size, 
                    validation_split = 0.15, callbacks=[earlystop])
endt = time.time()-start
print("Time for {} epochs: {:0.2f}ms".format(epochs,1000*endt))

model_cnn.evaluate(x_test_gs, y_test, verbose = 2)
model_mlp.evaluate(x_test_gs, y_test, verbose = 2)
