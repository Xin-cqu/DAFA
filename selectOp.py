# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
#import matplotlib.pyplot as plt
import h5py

dtype = torch.FloatTensor
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
#load_data=h5py.File("./fMRI_PC_SW/cam160.mat")
load_data=sio.loadmat("./fMRI_PC_SW/cam160.mat")
#double_lab=load_lab['lab'].astype(float)
# load_data=sio.loadmat("./fMRI_PC_SW/feature.mat")
#data_set=torch.FloatTensor(load_data['cam160'])#.cuda()
tempdata=load_data['cam160'].T
samples=(tempdata-np.min(tempdata))/(np.max(tempdata)-np.min(tempdata))
data_set=torch.FloatTensor(samples)#.cuda()
# data_set=torch.randn(4641,90).type(dtype)
# normalization of the input to the scale:[0,1]
# data_set[data_set.abs()<0.5]=0
#data_std=(data_set-data_set.min())/(data_set.max()-data_set.min())
#Todo lab need reformed
lab=torch.LongTensor(np.ones((412,1)))
tensor_dataset=TensorDataset(data_set,lab)
# x=Variable(data_set)
# y=Variable(lab)
print('The training size is: ',len(tensor_dataset))
train_loader=DataLoader(tensor_dataset,
                             batch_size=10,
                             shuffle=True,
                             num_workers=0
                             )
# for data, target in train_loader:
#      print(data,target)
#
EPOCH = 50
BATCH_SIZE = 412
LR = 0.0005
result=[]
# class MSEL1Loss(nn.Module):
#     def __init__(self,input,target):
#         self.input=input
#         self.target=target
#         return
#     def forward(self,input,target):
step=5
hidden=0
for i in range(200):
    hidden=(i+1)*step
    class AutoEncoder(nn.Module):
        def __init__(self):
            super(AutoEncoder, self).__init__()

            self.encoder = nn.Sequential(
                nn.Linear(12720, hidden),
                # nn.Sigmoid(),
                # nn.Linear(200, 20),
                nn.Sigmoid()
            )
            self.decoder = nn.Sequential(
                # nn.Linear(20, 200),
                #  nn.ReLU(),
                nn.Linear(hidden, 12720),
                nn.Tanh(),
                # compress to a range (0, 1)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            #encoded = nn.functional.dropout(encoded, p=0.6, training=self.training)
            decoded = self.decoder(encoded)
            return encoded, decoded
    autoencoder = AutoEncoder().cuda()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=1e-5)  # L2 norm
    # Todo: add l1 norm to implement the sparsity
    loss_func = nn.MSELoss()
    l1lambda = 0
    #encoded, decoded = autoencoder(Variable(data_set))
    # print(decoded)
    for epoch in range(EPOCH):
        for batch_idx, (x,y) in enumerate(train_loader):
            b_x = Variable(x, requires_grad=False)  # batch x, shape (batch, 51*4005)
            b_y = Variable(x, requires_grad=False)  # batch y, shape (batch, 51*4005)
            #b_label = Variable(y)  # batch label[0,1] and [-1,0]
            # print(batch_idx)
            encoded, decoded = autoencoder(b_y.cuda())
            # Todo: add l1 norm to implement the sparsity
            loss = loss_func(decoded, b_y.cuda()) #+ l1lambda * torch.abs(b_y).sum()  # mean square error with L1 norm
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # if ((batch_idx % 90 == 0) & (batch_idx > 0)):
            #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
    #result[i]=loss.item()
    print('Hidden nodes:',hidden,'| train loss: %.4f'%loss.item())  # print(encoded)
    # print('Save the encoder for feature selection.\n')
    # torch.save(autoencoder.encoder,'encoder.pkl')