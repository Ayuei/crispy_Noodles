import nn
import sys
import h5py
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data_dir = 'C:/Users/dsear/OneDrive/Uni/Deep Learning - COMP5329/Assignment 1/Code/crispy_Noodles-master/data/'

def onehot_labels(test_label):
    lbls = np.array(test_label)
    onehot = np.zeros((lbls.size, lbls.max()+1)) 
    onehot[np.arange(lbls.size), lbls] = 1   
    return onehot
    
def load_data(data_dir):
    print(data_dir)
    data = []
    label = []
    test = []

    with h5py.File(data_dir+'train_128.h5', 'r') as H:
        data = np.copy(H['data'])

    with h5py.File(data_dir+'train_label.h5', 'r') as H:
        label = np.copy(H['label'])

    with h5py.File(data_dir+'test_128.h5', 'r') as H:
        test = np.copy(H['data'])

    return data, label, test


def scale_data(data):
    X = np.array(data, dtype=np.float64)
    scaler = MinMaxScaler()
    scaler.fit(data)
    X = scaler.transform(data)
    return X

train, label, test = load_data(data_dir)

X = scale_data(train)
y = onehot_labels(label)

layers_dim = [128, 5, 3, 10]

#Based off the following implementation - https://github.com/pangolulu/neural-network-from-scratch
model = nn.Model(layers_dim)
model.train(X, y, num_passes=1000, epsilon=0.001, reg_lambda=0.01, print_loss=True)

predictions = model.predict(X[0:500])
lbls = np.array(label[0:500])
combined = np.stack((lbls, predictions), axis=-1)
predictionFrame = pd.DataFrame(combined, columns=['Actual', 'Predicted'])
accuracy = np.sum(lbls == predictions) / len(predictions)
print(accuracy)