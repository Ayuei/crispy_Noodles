import MLP
import json
import h5py
import numpy as np
import random

def accuracy(nn_, test_, lbs_):
    score = 0
    for x, label in zip(test_, lbs_):
        t = nn_.predict(x)
        dummy_var = np.argmax(nn_.predict(x))+1
        if label == np.argmax(nn_.predict(x))+1:
            score += 1

    return score/len(test)



def load_data(data_dir='data/'):
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
## Define how to input data
## How are they labelled etc
## How to split?

config = json.load(open('config.json', 'r'))

learning_rate = config['learning_rate']
epochs = config['epochs']
graph = config['graph']
data_dir = config['data_path']

nn = MLP.MLP([128, 64, 9])

train, label, test = load_data(data_dir)

indices = list(range(len(train)))
random.shuffle(indices)

train = list(map(train.__getitem__, indices))
label = list(map(label.__getitem__, indices))
test = train[50000:]
test_label = label[50000:]

MSE = nn.fit(train[0:50000], label[0:50000], learning_rate=learning_rate,
             epochs=epochs)

#Change to 9 output neurons, and get max of the neuron that activates?
#Need to change how everything works as it's no longer binary...
#argmax...

print('Accuracy is: '+str(accuracy(nn, test, test_label)))
