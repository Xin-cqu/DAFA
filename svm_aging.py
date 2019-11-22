from sklearn.model_selection import KFold
from sklearn import svm
import numpy as np
import logging
import scipy.io as sio

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

X_train,X_test=SplitData(412)

load_data=sio.loadmat("cam_yo.mat")
data_set=load_data['cam_yo'].T
lab=np.ones((412,1))
lab[0:220]=-1
for i in range(5):
    test_data = []
    train_data = []
    test_lab = []
    train_lab = []

    test_data=data_set[X_test[i]]
    test_lab=lab[X_test[i]]
    train_data=data_set[X_train[i]]
    train_lab=lab[X_train[i]]


    #print("test")
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(train_data, train_lab.ravel())

    print(clf.score(train_data, train_lab))
    y_hat = clf.predict(test_data)
    print(np.sum(y_hat==test_lab.T)/len(test_lab))