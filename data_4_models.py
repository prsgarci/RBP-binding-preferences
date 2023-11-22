import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def data_arrays(data_path):
    dataset = h5py.File(data_path, 'r')
    x_train = np.array(dataset['X_train']).transpose([0,2,1])
    y_train = np.array(dataset['Y_train'])
    x_valid = np.array(dataset['X_valid']).transpose([0,2,1])
    y_valid = np.array(dataset['Y_valid'])
    x_test = np.array(dataset['X_test']).transpose([0,2,1])
    y_test = np.array(dataset['Y_test'])
    alphabet = 'ACGT'

    
    return x_train, y_train, x_valid, y_valid, x_test, y_test, alphabet


def input_shape_check(x_train, x_test,x_valid):
    if x_train.shape[-1] == 4:
        print("Input shape is correct: " + x_train.shape)
    else:
        # Adjust input shape
        x_train = x_train[:,:,:4]
        x_test = x_test[:,:,:4]
        x_valid = x_valid[:,:,:4]

        # Print the shape of the training data
        print("Input shape adjusted:")
        print(x_train.shape)
        print(x_test.shape)
        print(x_valid.shape)
    return x_train, x_test, x_valid