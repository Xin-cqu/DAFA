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

#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
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
load_data=sio.loadmat("./fMRI_PC_SW/data.mat")
data_set=torch.FloatTensor(load_data['data'].T)#.cuda()
#load_test=h5py.File("./fMRI_PC_SW/Cam_brainNet_middle.mat")
#test_data=torch.FloatTensor(load_test['Cam_brainNet_middle'])
#middle_data=h5py.File("./fMRI_PC_SW/Cam_brainNet_middle.mat")
#middledata=torch.FloatTensor(middle_data['Cam_brainNet_middle'])
lab=torch.LongTensor(np.ones((15028)))
lab[0:7514]=0
# middle_lab=torch.LongTensor(np.zeros((44944)))
predict_label=np.zeros((221,68))
# predict_mlabel=np.zeros((212,212))
# split into 396 loops for each person
#load_lab=sio.loadmat("./fMRI_PC_SW/lab.mat")
#lab=torch.FloatTensor(load_lab['lab'])#.cuda()

for i in range(68):
    test_data = data_set[i*221:(i+1)*221, :]
    test_lab = lab[i*221:(i+1)*221]

    if i==0:
        train_data = data_set[(i+1)*221:15028, :]
        train_lab = lab[(i + 1) * 221:15028]
    elif i==67:
        train_data = data_set[0:(i + 1) * 221, :]
        train_lab = lab[0:(i + 1) * 221]
    else:
        train_data=torch.cat((data_set[0:i*221,:],data_set[(i+1)*221:15028,:]),0)
        train_lab = torch.cat( (lab[0:i * 221], lab[(i + 1) * 221:15028]),0)

    #pdb.set_trace()
    tensor_dataset = TensorDataset(train_data, train_lab)
    test_dataset = TensorDataset(test_data, test_lab)
    #middle_dataset= TensorDataset(middledata,middle_lab)
    print('The training size is: ', len(tensor_dataset))
    train_loader = DataLoader(tensor_dataset,
                              batch_size=221,
                              shuffle=True,
                              num_workers=0
                              )

    test_loader = DataLoader(test_dataset,
                             batch_size=221,
                             shuffle=False,
                             num_workers=0
                             )
    # middle_loader = DataLoader(middle_dataset,
    #                            batch_size=212,
    #                            shuffle=False,
    #                            num_workers=0
    #                            )

    EPOCH = 1
    BATCH_SIZE = 68
    LR = 0.00015


    class ClassfyNet(nn.Module):
        def __init__(self):
            super(ClassfyNet, self).__init__()
            #self.l1 = nn.Linear(6670, 180)
            self.l2 = nn.Linear(91, 50)
            self.l3 = nn.Linear(50, 10)
            self.l4 = nn.Linear(10, 2)

        def forward(self, x):
            #x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            return F.log_softmax(self.l4(x), dim=1)  # problem
            # return self.l5(x)


    classfyNet = ClassfyNet().to(device)
    # pretrain_model = torch.load('aging.pkl')
    # test = classfyNet
    # model_dict = classfyNet.state_dict()
    # predict = pretrain_model.state_dict()
    # model_dict['l1.weight'] = predict['0.weight']
    # model_dict['l1.bias'] = predict['0.bias']
    # classfyNet.load_state_dict(model_dict)
    #Todo
    #classfyNet.parameters()[0].requires_grad()
    # pdb.set_trace()
    optimizer = torch.optim.Adam(classfyNet.parameters(), lr=LR, weight_decay=1e-7)  # L2 norm
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
            if batch_idx % 220 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


    def test():
        with torch.no_grad():
            test_loss = 0
            correct = 0
            # 测试集
            for data, target in test_loader:
                data, target = Variable(data), Variable(target)
                output = classfyNet(data.to(device))
                # sum up batch loss
                test_loss += loss_func(output, target.to(device)).item()
                # get the index of the max
                # pdb.set_trace()
                pred = output.data.max(1, keepdim=True)[1].transpose(0, 1)
                predict_label[:,i]=pred.cpu().data.numpy()
                correct += target.data.eq(pred.cpu()).sum()

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
#np.savetxt('aflabel.txt',predict_label)
#np.savetxt('mplabel.txt',predict_mlabel)


