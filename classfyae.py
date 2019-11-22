# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import h5py
import pdb
import torch.nn.functional as F
from sklearn.model_selection import KFold
import numpy as np
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
logger.info("Using: "+device.type)
dtype = torch.cuda.FloatTensor
class TensorDataset(Dataset):
    '''
    TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    '''


    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
'''
Load data from Matlab: PC of the slice-windows of fMRI(45mci-46hc), window's length is 30 time point.
One person would construct 51 windows, so the batch size is 51 and every 91 batches is an epoch standing for the whole data
'''
logger.info("Loading data")
load_data=sio.loadmat("./fMRI_PC_SW/camyo.mat")
data_set=torch.FloatTensor(load_data['camyo'])#.cuda()
#data_set=data_set[0:412,:]
#idx=torch.LongTenser([0:1:87344])
#data_set=torch.index_select(data_set,0,idx)

#load_test=h5py.File("./fMRI_PC_SW/Cam_brainNet_middle.mat")
#test_data=torch.FloatTensor(load_test['Cam_brainNet_middle'])
#middle_data=h5py.File("./fMRI_PC_SW/Cam_brainNet_middle.mat")
#middledata=torch.FloatTensor(middle_data['Cam_brainNet_middle'])
lab=torch.LongTensor(np.ones((412)))
lab[220:412]=0
#middle_lab=torch.LongTensor(np.zeros((44944)))
predict_label=np.zeros((1,412))
#predict_mlabel=np.zeros((212,212))
# split into 396 loops for each person
logger.info("Data preprocessed")
for i in range(5):
    logger.info("Starting 5 fold cross validation:"+str(i))
    test_data = []
    train_data = []
    test_lab=[]
    train_lab=[]
    #notest = 0
    for j in range(len(X_test[i])):
        if j==0:
            test_data=data_set[X_test[i][j]:(1+X_test[i][j]),:]
            test_lab=lab[X_test[i][j]:(1+X_test[i][j])]
        else:
            test_lab = torch.cat((test_lab, lab[X_test[i][j]:(1 + X_test[i][j]) ]), 0)
            test_data = torch.cat((test_data, data_set[X_test[i][j] :(1 + X_test[i][j]) , :]), 0)
        #test_data=torch.cat((test_data,data_set[X_test[i][j]*212:(1+X_test[i][j])*212,:]),0)
        #train_data=torch.cat((train_data,data_set[X_train[i][j]*212:(1+X_train[i][j])*212,:]),0)
        #test_lab = torch.cat((test_lab, lab[X_test[i][j] * 212:(1 + X_test[i][j]) * 212, :]), 0)
        #train_lab=torch.cat((train_lab,lab[X_train[i][j]*212:(1+X_train[i][j])*212,:]),0)
    for l in range(len(X_train[i])):
        if l==0:
            train_data=data_set[X_train[i][l]:(1+X_train[i][l]),:]
            train_lab=lab[X_train[i][l]:(1+X_train[i][l])]
        else:
            train_data = torch.cat((train_data, data_set[X_train[i][l] :(1 + X_train[i][l]), :]), 0)
            train_lab = torch.cat((train_lab, lab[X_train[i][l] :(1 + X_train[i][l]) ]), 0)
    # test_data=data_set[X_test[i]]
    # test_data = data_set[i*212:(i+1)*212, :]
    # test_lab = lab[i*212:(i+1)*212]
    #
    # if i==0:
    #     train_data = data_set[(i+1)*212:83952, :]
    #     train_lab = lab[(i + 1) * 212:83952]
    # elif i==395:
    #     train_data = data_set[0:(i + 1) * 212, :]
    #     train_lab = lab[0:(i + 1) * 212]
    # else:
    #     train_data=torch.cat((data_set[0:i*212,:],data_set[(i+1)*212:83952,:]),0)
    #     train_lab = torch.cat( (lab[0:i * 212], lab[(i + 1) * 212:83952]),0)

    #pdb.set_trace()
    tensor_dataset = TensorDataset(train_data, train_lab)
    test_dataset = TensorDataset(test_data, test_lab)
    #middle_dataset= TensorDataset(middledata,middle_lab)
    print('The training size is: ', len(tensor_dataset))
    train_loader = DataLoader(tensor_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0
                              )

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0
                             )
    # middle_loader = DataLoader(middle_dataset,
    #                            batch_size=212,
    #                            shuffle=False,
    #                            num_workers=0
    #                            )

    EPOCH = 10
    BATCH_SIZE = 412
    LR = 0.0015


    class ClassfyNet(nn.Module):
        def __init__(self):
            super(ClassfyNet, self).__init__()
            self.l1 = nn.Linear(740, 100)
            self.l2 = nn.Linear(100, 50)
            self.l3 = nn.Linear(50, 10)
            self.l4 = nn.Linear(10, 2)

        def forward(self, x):
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            return F.log_softmax(self.l4(x), dim=1)  # problem
            #return self.l4(x)


    classfyNet = ClassfyNet().to(device)
    optimizer = torch.optim.Adam(classfyNet.parameters(), lr=LR, weight_decay=1e-5)  # L2 norm
    loss_func = nn.NLLLoss()


    def train(epoch):
        # 每次输入barch_idx个数据
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = classfyNet(data.to(device))
            # loss
            # pdb.set_trace()
            loss = loss_func(output, target.to(device))
            loss.backward()
            # update
            optimizer.step()
            if batch_idx % 329 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


    def test():
        with torch.no_grad():
            test_loss = 0
            correct = 0
            notest = 0
            # 测试集
            for data, target in test_loader:
                data, target = Variable(data), Variable(target)
                output = classfyNet(data.to(device))
                # sum up batch loss
                test_loss += loss_func(output, target.to(device)).item()
                # get the index of the max
                #pdb.set_trace()
                pred=torch.max(output, 1)[1]#.transpose(0,1)
                #pred = output.data.max(1, keepdim=True)[1].transpose(0, 1)
                predict_label[:,i*len(test_loader)+notest]=pred.cpu().data.numpy()
                correct += pred.cpu().eq(target.data.view_as(pred)).sum()
                notest+=1
            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    # def middle_test():
    #     j=0
    #     with torch.no_grad():
    #         middle_loss=0
    #         correct=0
    #         for data, target in middle_loader:
    #             data, target = Variable(data), Variable(target)
    #             output = classfyNet(data.to(device))
    #             # sum up batch loss
    #             middle_loss += loss_func(output, target.to(device)).data[0]
    #             # get the index of the max
    #             # pdb.set_trace()
    #             pred = output.data.max(1, keepdim=True)[1].transpose(0, 1)
    #             #pdb.set_trace()
    #             predict_mlabel[:,j]=predict_mlabel[:,j]+pred.cpu().data.numpy()
    #             correct += target.data.eq(pred.cpu()).sum()
    #             j=j+1
    #         middle_loss /= len(middle_loader.dataset)
    #         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #             middle_loss, correct, len(middle_loader.dataset),
    #             100. * correct / len(middle_loader.dataset)))
    for epoch in range(EPOCH):
        train(epoch)
        test()
        #middle_test()
np.savetxt('5KFaelabel.txt',predict_label)
logger.info("Done.")
#np.savetxt('mplabel.txt',predict_mlabel)


