
import numpy as np
import torch
from torch.autograd import Variable
import h5py
import scipy.io as sio
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def getFeatures(inputData):
    #Use the AE training parameters to extract feature vector
    encoder=torch.load('aging_160.pkl').to(device)
    encoded=encoder(inputData)
    return encoded

# load data
#device='cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'
logger.info("Start print log")
logger.info("Begaining:")
load_data=h5py.File("./fMRI_PC_SW/cam160sl.mat")
logger.info("Load data.")
data_set=torch.FloatTensor(load_data['cam160']).to(device)#.cuda()
logger.info("Load data done.")
X=Variable(data_set)
encoded=getFeatures(X)
logger.info("Calculation done.")
features=encoded.data
#np.savetxt('aging_yvo.txt',features.cpu().numpy())
sio.savemat('aeyo160.mat',{'aeyo160':features.numpy()})
logger.info("features saved!")