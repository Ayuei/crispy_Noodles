import MLP
import json
import h5py
import numpy as np

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

nn = MLP.MLP([128,1], 'relu')

train, label, test = load_data(data_dir)

MSE = nn.fit(train, test, learning_rate=learning_rate,
             epochs=epochs)

print(MSE[len(MSE)-10:-1])
print(MSE[-1])
