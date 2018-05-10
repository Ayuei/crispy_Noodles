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

def getPredFrameAndScore(truth, preds):
    combined = np.stack((truth, preds), axis=-1)
    predictionFrame = pd.DataFrame(combined, columns=['Actual', 'Predicted'])
    accuracy = np.sum(truth == preds) / len(preds)
    return predictionFrame, accuracy
   
def generatePredictionScore(model, X, Y):
    predictions = model.predict(X)
    return getPredFrameAndScore(Y, predictions)
    
train, label, test = load_data(data_dir)

X = scale_data(train)
y = onehot_labels(label)

Xtrain = X[:59000]
Ytrain = y[:59000]

Xtest = X[59000:60000] #unseen
Ytest = np.array(label)[59000:60000] #unseen

layers_dim = [128, 64, 32, 16, 8, 8, 10]

model = nn.Model(layers_dim)
model.train(Xtrain, Ytrain, num_passes=3001, epsilon=0.0002, reg_lambda=0.01, print_loss=True)

testUnseenFrame, testUnseenAcc = generatePredictionScore(model, Xtest, Ytest)

#Review it's working by testing a subset of the train data
testSeenFrame, testSeenAcc = generatePredictionScore(model, Xtrain[:5000], np.array(label)[:5000])
