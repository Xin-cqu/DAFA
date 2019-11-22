# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import h5py
import pdb


torch.manual_seed(1)
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
Load data from Matlab: PC of the slice-windows of fMRI(213young-183old), window's length is 50 time point.
One person would construct 212 windows, so the batch size is 212 and every 396 batches is an epoch standing for the whole data
'''
load_data=sio.loadmat("./fMRI_PC_SW/cam.mat")
data_set=torch.FloatTensor(load_data['cam'].T)#.cuda()

lab=torch.FloatTensor(np.ones((12720,1)))

tensor_dataset=TensorDataset(data_set, lab)

print('The training size is: ',len(tensor_dataset))
train_loader=DataLoader(tensor_dataset,
                             batch_size=76,
                             shuffle=True,
                             num_workers=0
                             )

EPOCH = 20
LR = 0.002
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(608, 100),
            nn.Sigmoid(),

        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 608),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder().cuda()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR,weight_decay=1e-5)#1e-6) # L2 norm
#Todo: add l1 norm to implement the sparsity
loss_func = nn.MSELoss()
l1lambda=0
sparsity_level=0.05

for epoch in range(EPOCH):
    for batch_idx, (x, y) in enumerate(train_loader):
        b_x = Variable(x).cuda()   # batch x, shape (batch, 51*4005)
        b_y = Variable(x).cuda()   # batch y, shape (batch, 51*4005)
        # b_label = Variable(y)               # batch label[0,1] and [-1,0]
        # print(batch_idx)
        encoded, decoded = autoencoder(b_y)
        # Todo: add l1 norm to implement the sparsity
        #klreg=torch.sum(sparsity_level*torch.log(sparsity_level/torch.mean(decoded.data,0))+(1-sparsity_level)*torch.log((1-sparsity_level)/(1-torch.mean(decoded.data,0))))
        loss = loss_func(decoded, b_y)#+0.000005*klreg  #+nn.KLDivLoss()mean square error with L1 norm
        #pdb.set_trace()
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        #if ((batch_idx % 211 == 0) & (batch_idx>0)):
    print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
# print(loss.data[0])            # print(encoded)
print('Save the encoder for feature selection.\n')
torch.save(autoencoder.encoder,'aging_ndim.pkl')
