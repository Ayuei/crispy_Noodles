from sequential import Sequential
from sklearn.preprocessing import MinMaxScaler
import h5py
import json
import numpy as np
import random
from layers import *
import pickle

epochs = 100

def accuracy_per_class(nn_, test_, lbs_):
    classes_score = np.array([0 for i in range(len(lbs_[0]))])
    classes_counts = np.array([0 for i in range(len(lbs_[0]))])

    for x, lbl in zip(test_, lbs_):
        if np.argmax(lbl) == np.argmax(nn_.predict(x)):
            classes_score[np.argmax(lbl)] += 1
        classes_counts[np.argmax(lbl)] += 1

    return classes_score/classes_counts


def raw_accuracy(test_, lbs_):
    score = 0
    for x, lbl in zip(test_, lbs_):
        if np.argmax(lbl) == np.argmax(x):
            score += 1
    return score/len(test_)


def accuracy(nn_, test_, lbs_):
    score = 0
    for x, lbl in zip(test_, lbs_):
        if np.argmax(lbl) == np.argmax(nn_.predict(x)):
            score += 1

    return score/len(test_)


# Returns onehot labels for multiclass classification
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
#epochs = config['epochs']
graph = config['graph']
data_dir = config['data_path']
load_pre_trained = False

learning_rate = 0.001

train, label, test = load_data(data_dir)

def model_adam(X, y, verbose):
    nn = Sequential(learning_rate=learning_rate, epochs=epochs, batch_size=100,
                    learning_rate_decay=0.95, weight_decay=0.01)

    nn.add(Dense(n=200, in_shape=X.shape[1]))
    nn.add(BatchNorm())
    nn.add(Dense(n=100))
    nn.add(BatchNorm())
    nn.add(Dense(n=80))
    nn.add(BatchNorm())
    nn.add(Dense(n=40))
    nn.add(BatchNorm())
    nn.add(Dense(n=80))
    nn.add(BatchNorm())
    nn.add(Dense(n=100))
    nn.add(BatchNorm())
    nn.add(Dense(n=200))
    nn.add(BatchNorm())
    nn.add(Dense(n=10, activation="softmax"))
    nn.compile(loss="cross_entropy_softmax", optimiser="Adam")

    nn.fit(X, y, verbose)

    return nn

def model_adam_dropout(X, y, verbose):
    nn = Sequential(learning_rate=learning_rate*0.1, epochs=epochs, batch_size=100,
                    learning_rate_decay=0.95, weight_decay=0.01)

    nn.add(Dense(n=200, in_shape=X.shape[1]))
    nn.add(Dropout(0.2))
    nn.add(Dense(n=100))
    nn.add(Dropout(0.2))
    nn.add(Dense(n=80))
    nn.add(Dropout(0.2))
    nn.add(Dense(n=40))
    nn.add(Dropout(0.2))
    nn.add(Dense(n=80))
    nn.add(Dropout(0.2))
    nn.add(Dense(n=100))
    nn.add(Dropout(0.2))
    nn.add(Dense(n=200))
    nn.add(Dropout(0.2))
    nn.add(Dense(n=10, activation="softmax"))
    nn.compile(loss="cross_entropy_softmax", optimiser="Adam")

    nn.fit(X, y, verbose)

    return nn

def model_SGD(X, y, verbose):
    nn = Sequential(learning_rate=learning_rate*0.5, epochs=epochs, batch_size=100,
                    learning_rate_decay=0.95, weight_decay=0.01)
    nn.add(Dense(n=200, in_shape=X.shape[1]))
    nn.add(BatchNorm())
    nn.add(Dense(n=100))
    nn.add(BatchNorm())
    nn.add(Dense(n=80))
    nn.add(BatchNorm())
    nn.add(Dense(n=40))
    nn.add(BatchNorm())
    nn.add(Dense(n=80))
    nn.add(BatchNorm())
    nn.add(Dense(n=100))
    nn.add(BatchNorm())
    nn.add(Dense(n=200))
    nn.add(BatchNorm())
    nn.add(Dense(n=10, activation="softmax"))
    nn.compile(loss="cross_entropy_softmax", optimiser="SGD")

    nn.fit(X, y, verbose)

    return nn

indices = list(range(len(train)))
random.shuffle(indices)

train = list(map(train.__getitem__, indices))
label = list(map(label.__getitem__, indices))

X = scale_data(train)
y = np.array(onehot_labels(label), dtype=np.float64)

train_set = X[0:48000], y[0:48000]

ensemble = []

if load_pre_trained:
    ensemble = [model for model in [model_adam(train_set[0], train_set[1], True),
                                    model_adam_dropout(train_set[0], train_set[1], True),
                                    model_SGD(train_set[0], train_set[1], True)]]
else:
    import glob
    for model in glob.glob('trained_models/'):
        ensemble.append(pickle.load(open(model, 'rb')))

validation_set = X[48000:50000], y[48000:50000]
test_set = X[50000:], y[50000:]

weights = [np.zeros(np.shape(y[0])) for i in range(len(ensemble))]

accs = []

for model in ensemble:
    print('Accuracy: ', accuracy(model, test_set[0], test_set[1]))
    acc = accuracy_per_class(model, validation_set[0], validation_set[1])
    accs.append(acc)

weights = accs/np.sum(accs, axis=0)

predicted_softmaxes = [[] for i in range(len(ensemble))]

for test_example in test_set[0]:
    for i, model in enumerate(ensemble):
        predicted_softmaxes[i].append(model.predict(test_example))

preds = np.squeeze(np.array(predicted_softmaxes))

preds_weighted = []

for i, pred in enumerate(preds.T):
    weighted_pred = pred * weights.T[i]

    preds_weighted.append(np.sum(weighted_pred, axis=1))

preds = np.sum(preds, axis=0)/len(ensemble)

preds_weighted = np.array(preds_weighted).T

model_names = ["adam", "adam_dropout", "sgdmomentum"]

for model, name in zip(ensemble, model_names):
    pickle.dump(model, open('trained_models/'+name+".model", 'wb+'))

print('Non-weighted:', raw_accuracy(preds, test_set[1]))
print('Weighted:', raw_accuracy(preds_weighted, test_set[1]))
