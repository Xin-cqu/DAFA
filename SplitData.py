from sklearn.model_selection import KFold
import numpy as np
def SplitData(samples):
    kf=KFold(n_splits=5,shuffle=True,random_state=0)
    X=np.arange(0,samples,1)
    i=0
    X_train={}
    X_test={}
    for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
        X_train[i], X_test[i] = X[train_index], X[test_index]
        i=i+1
    return X_train,X_test

X_train,X_test=SplitData(415)
#np.arange(0,415,1),np.row_stack((np.ones((221,1)),np.zeros((194,1)))),random_state=0