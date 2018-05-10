from sequential import Sequential
from sklearn.preprocessing import MinMaxScaler
import h5py
import json
import numpy as np
import random
from layers import *
#from keras_test import main

def accuracy(nn_, test_, lbs_):
    score = 0
    for x, label in zip(test_, lbs_):
        t = nn_.predict(x)
        dummy_var = np.argmax(nn_.predict(x))
        if label == np.argmax(nn_.predict(x)):
            score += 1

    return score/len(test_)

#Returns onehot labels for multiclass classification
def onehot_labels(test_label):
    lbls = np.array(test_label)
    onehot = np.zeros((lbls.size, lbls.max()+1)) 
    onehot[np.arange(lbls.size), lbls] = 1   
    return onehot
    
def load_data(data_dir='data/'):
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

config = json.load(open('config.json', 'r'))

learning_rate = config['learning_rate']
epochs = config['epochs']
graph = config['graph']
data_dir = config['data_path']

learning_rate = 0.001

nn = Sequential(learning_rate=learning_rate, epochs=50)

train, label, test = load_data(data_dir)

nn.add(Dense(n=60, in_shape=train.shape[1]))
nn.add(Dense(n=100))
nn.add(Dropout(0.3))
nn.add(Dense(n=100))
nn.add(Dropout(0.3))
nn.add(Dense(n=100))
nn.add(Dropout(0.3))
nn.add(Dense(n=100))
nn.add(Dense(n=10, activation="softmax"))
nn.compile(loss="cross_entropy_softmax")

indices = list(range(len(train)))
random.shuffle(indices)

train = list(map(train.__getitem__, indices))
label = list(map(label.__getitem__, indices))

X = scale_data(train)
y = np.array(onehot_labels(label), dtype=np.float64)

nn.fit(X[0:50000], y[0:50000])

print('Accuracy is: '+str(accuracy(nn, X[50000:], label[50000:])))
