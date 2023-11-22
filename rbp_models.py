import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import models, layers, callbacks, activations, optimizers


'''DeepBind Model'''

def deepbind():
    #Build the model
    model = models.Sequential()
    #layer1
    model.add(layers.InputLayer(input_shape=(200, 4))) # 4 channel input
    #layer2
    model.add(layers.Conv1D(filters=16, kernel_size=24, padding='same'))
    model.add(layers.Activation(activations.relu))
    #layer3
    model.add(layers.MaxPooling1D(pool_size=25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(units=32, activation='relu')) #model says "one hidden layer with 32 ReLu units"?
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(units=1, activation='linear'))
    model.add(layers.Activation('sigmoid'))
    
    
    #print(model.summary())
    return model



'''DeepBind with Exponential activation'''
def deepbind_exp():
    tf.keras.backend.clear_session()
    #Build the model
    model = models.Sequential()
    #layer1
    model.add(layers.InputLayer(input_shape=(200, 4))) # 4 channel input
    #layer2
    model.add(layers.Conv1D(filters=16, kernel_size=24, padding='same'))
    model.add(layers.Activation(activations.exponential))
    #layer3
    model.add(layers.MaxPooling1D(pool_size=25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(units=32, activation='relu')) #model says "one hidden layer with 32 ReLu units"?
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(units=1, activation='linear'))
    model.add(layers.Activation('sigmoid'))
    
    
    #print(model.summary())
    
    return model

##################################################################################################################
##################################################################################################################

''' Representation Learning Paper model'''
def baseline_cnn():
    
    #Build the model
    tf.keras.backend.clear_session()
    model = models.Sequential()
    # layer1
    model.add(layers.InputLayer(input_shape=(200, 4))) # 4 channel input
    
    # layer2
    l2_regularizer = tf.keras.regularizers.L2(1e-6)
    model.add(layers.Conv1D(filters=30, kernel_size=19, strides=1, padding='same', kernel_regularizer=l2_regularizer))
    # add batch normalization
    # batch_norm = tf.keras.layers.BatchNormalization()
    model.add(layers.Activation(activations.relu))

    # layer3
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Conv1D(filters=30, kernel_size=19, strides=1, padding='same'))
    # model.add(layers.Conv1D(filters=30, kernel_size=19, strides= 1, padding='same', kernel_regularizer=l2_regularizer))
    model.add(layers.MaxPooling1D(pool_size=50, strides=50))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    
    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(units=1, activation='linear'))
    model.add(layers.Activation('sigmoid'))

    # model.summary()    
    return model


''' Representation Learning Paper model with Exponential activation'''
def baseline_cnn_exp():
    
    tf.keras.backend.clear_session()
    #Build the model
    tf.keras.backend.clear_session()
    model = models.Sequential()
    # layer1
    model.add(layers.InputLayer(input_shape=(200, 4))) # 4 channel input
    
    # layer2
    l2_regularizer = tf.keras.regularizers.L2(1e-6)
    model.add(layers.Conv1D(filters=30, kernel_size=19, strides=1, padding='same', kernel_regularizer=l2_regularizer))
    # add batch normalization
    # batch_norm = tf.keras.layers.BatchNormalization()
    model.add(layers.Activation(activations.exponential))

    # layer3
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Conv1D(filters=30, kernel_size=19, strides=1, padding='same'))
    # model.add(layers.Conv1D(filters=30, kernel_size=19, strides= 1, padding='same', kernel_regularizer=l2_regularizer))
    model.add(layers.MaxPooling1D(pool_size=50, strides=50))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    
    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(units=1, activation='linear'))
    model.add(layers.Activation('sigmoid'))
    # model.summary()

    
    return model