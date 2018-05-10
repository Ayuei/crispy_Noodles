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
        #t = nn_.predict(x)
        #dummy_var = np.argmax(nn_.predict(x))
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
   
def generatePredictionScore(nn_, X, Y):
    preds = np.argmax(nn_.predict(X), axis=1)
    combined = np.stack((Y, preds), axis=-1)
    predictionFrame = pd.DataFrame(combined, columns=['Actual', 'Predicted'])
    accuracy = np.sum(Y == preds) / len(preds)
    confMatrix = pd.crosstab(predictionFrame['Actual'], predictionFrame['Predicted'])
    return predictionFrame, confMatrix, accuracy

def testAcrossFolds(X, Y, y_oh, nn):
    xtr,ytr,yohtr,xt,yt = kfolds.Kfolds().kFoldData(X, y, y_oh, kfolds=5)
    predictions = []
    confMatrices = []
    accuracies = []
    
    fold = 1
    for xtrain, ytrain, y_oh_train, xtest, ytest in zip(xtr,ytr,yohtr,xt,yt):
        fold_start = time.time()
        print('Train Size:%s | Test Size:%s' % (np.shape(xtrain)[0], np.shape(xtest)[0]))
        nn.fit(xtrain, y_oh_train, verbose=True)
        frame, cfm, acc = generatePredictionScore(nn, xtest, ytest)
        predictions.append(frame)
        accuracies.append(acc)
        confMatrices.append(cfm)
        print("\n------Fold: %s | Accuracy:%s | Time taken: %s seconds ------\n" % (fold, acc, np.round(time.time() - fold_start, 2)))
        fold+=1
        
    return accuracies, predictions, confusionMatrices
    
def writeFinalPredictions(predictions):
    #Expects numpy array of predictions
    h5f = h5py.File('Predicted_labels.h5', 'w')
    h5f.create_dataset('Predicted_labels', data=predictions)
    h5f.close()
    
config = json.load(open('config.json', 'r'))

learning_rate = config['learning_rate']
epochs = config['epochs']
graph = config['graph']
data_dir = config['data_path']

learning_rate = 0.001

nn = Sequential(learning_rate=learning_rate, epochs=10, batch_size=100)

train, label, test = load_data(data_dir)

nn.add(Dense(n=80, in_shape=train.shape[1]))
nn.add(Batch_norm())
nn.add(Dense(n=100))
nn.add(Batch_norm())
nn.add(Dense(n=100)) #50
nn.add(Batch_norm())
nn.add(Dense(n=100))
nn.add(Batch_norm())
nn.add(Dense(n=100))
nn.add(Batch_norm())
nn.add(Dense(n=10, activation="softmax"))
nn.compile(loss="cross_entropy_softmax", optimiser="adam")


X = scale_data(train)
y = np.array(label)
y_oh = onehot_labels(label)

#K-fold validation of model
testAcrossFolds(X, y, y_oh, nn)
    
X_Test_Final = scale_data(test)
nn.fit(X, y_oh, verbose=True)
final_preds = np.argmax(nn.predict(X_Test_Final), axis=1)
writeFinalPredictions(final_preds)


