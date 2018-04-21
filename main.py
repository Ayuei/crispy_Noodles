import MLP
import json
import h5py
import numpy as np
import random

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

nn = MLP.MLP([128, 64, 1], 'relu')

train, label, test = load_data(data_dir)

indices = list(range(len(train)))
indices = random.shuffle(indices)

train = map(train.__getitem__, indices)
label = map(label.__getitem__, indices)
test = train[1001:2000]
test_label = label[1001:2000]

MSE = nn.fit(train[0:1000], label[0:1000], learning_rate=learning_rate,
             epochs=epochs)

#Change to 9 output neurons, and get max of the neuron that activates?
#Need to change how everything works as it's no longer binary...
#argmax...

print(MSE[len(MSE)-10:-1])
print(MSE[-1])
