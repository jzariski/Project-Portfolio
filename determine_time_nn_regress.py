import os
import numpy as np
import sys
import time
import random
import scipy
import tensorflow
from pathlib import Path
#import matplotlib.pyplot as plt
import statistics as s
#import matplotlib.colors as colors
import numpy.linalg as lg
import h5py
from keras import layers
from tensorflow import keras
sys.path.append('../shocktubecalc')


def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        train_out = np.array(f["labels_train"])
        train_in = np.array(f["inputs_train"])
        dev_out = np.array(f["labels_dev"])
        dev_in = np.array(f["inputs_dev"])
        test_out = np.array(f["labels_test"])
        test_in = np.array(f["inputs_test"])
    return train_in, train_out, dev_in, dev_out, test_in, test_out
    
"""    
def oneHotEncode(outputs, dimsi, dimsj):
    parts = np.linspace(0,0.25,dimsj)
    myDict = {}
    for i in range(0, len(parts)):
        myDict[parts[i]] = i
    
    finaloutputs = np.zeros((dimsi, dimsj))
    for i in range(0,dimsi):
        index = myDict[outputs[i]]
        finaloutputs[i,index] = 1
    
    return finaloutputs
"""
     
    
    
    
def create_nn(input_shape: tuple, n_outputs: int) -> tensorflow.keras.models.Model:

    model = keras.Sequential()
    model.add(tensorflow.keras.Input(shape=(None, 500)))
    #model.add(layers.Bidirectional((layers.GRU(50))))
    #model.add(layers.Flatten())
    model.add(layers.Dense(units=512, activation='linear'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=512, activation='linear'))
    model.add(layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dense(units=1, activation='linear'))
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-3),loss='mse')
    return model
    


train_in, train_out, dev_in, dev_out, test_in, test_out = load_hdf5("datasethigh_noise.hdf5")
train_in2, train_out2, dev_in2, dev_out2, test_in2, test_out2 = load_hdf5("datasethigh.hdf5")

#outputs = oneHotEncode(train_out,train_in.shape[0], 50)
es = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=10)
#mcp_save = tensorflow.keras.callbacks.ModelCheckpoint('modelcheck.hdf5', save_best_only=True, monitor='loss', mode='min')
reduce_lr_loss = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=8, verbose=1, epsilon=1e-4, mode='min')



deep = create_nn(train_in.shape, 50)
deep.summary()

deep.fit(train_in, train_out, verbose=1, epochs=200, batch_size=16,callbacks=[es, reduce_lr_loss])
deep.fit(dev_in2, dev_out2, verbose=1, epochs=200, batch_size=16,callbacks=[es, reduce_lr_loss])
deep.fit(dev_in, dev_out, verbose=1, epochs=200, batch_size=16,callbacks=[es, reduce_lr_loss])
deep.fit(train_in2, train_out2, verbose=1, epochs=200, batch_size=16,callbacks=[es, reduce_lr_loss])




deep.save('modeltrainedhigh_both.h5')

#deep.fit(np.expand_dims(train_in, axis=-1), train_out, verbose=1, epochs=1, batch_size=16)
#deep.summary()
train_in3, train_out3, dev_in3, dev_out3, test_in3, test_out3 = load_hdf5("datasetlow.hdf5")
preds = deep.predict(train_in3)
error = 0
for i in range(len(preds)):
    error += abs(preds[i] - train_out3[i])
print(error / len(preds))
    




    





