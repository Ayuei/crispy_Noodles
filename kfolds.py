import numpy as np

class Kfolds:
    def kFoldIndexes(self, X, kfolds):
        Nrows = len(X)
        all_row_indexes = np.arange(Nrows)
        np.random.shuffle(all_row_indexes)
        setLengths = int(np.ceil(Nrows/kfolds))
        testSetIndexes = [all_row_indexes[i:i + setLengths] for i in range(0, Nrows, setLengths)]
        xtrainIdxs = []
        xtestIdxs = []
        
        all_row_indexes = np.sort(all_row_indexes)
        
        for ts in testSetIndexes:
            xtestIdxs.append(ts)
            xns = np.delete(all_row_indexes, ts)
            xtrainIdxs.append(xns)
    
        return xtrainIdxs, xtestIdxs
    
    #Split data up according to given indexes
    def kFoldData(self, X, y, y_oh, kfolds=10):
        
        train_ixs, test_ixs = self.kFoldIndexes(X, kfolds)
        
        xtrain = []
        ytrain = []
        y_oh_train = []
        
        xtest = []
        ytest = []
    
        for train_ix, test_ix in zip(train_ixs, test_ixs):
            xtrain.append(X[train_ix][:])
            ytrain.append(np.take(y, train_ix))
            y_oh_train.append(y_oh[train_ix][:])
                    
            xtest.append(X[test_ix][:])
            ytest.append(np.take(y, test_ix))
                
        return xtrain, ytrain, y_oh_train, xtest, ytest