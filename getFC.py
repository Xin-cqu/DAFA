import torch
import numpy as np
import scipy.io as sio
def getFeatures(inputData):
    #Use the SAE training parameters to extract feature vector
    encoder=torch.load('aging_160.pkl')
    encoded=encoder(inputData)
    return encoded

# Get W for encoder
model=torch.load('aging_160.pkl')
print(model)
params=model.state_dict()
for k,v in params.items():
    print(k)
# print(params['0.weight'])
# print(params['0.bias'])
weights=params['0.weight'].cpu().numpy()
#print(np.size(weights.reshape(180,6670)))
#print(weights.size())
sio.savemat('weights_160_sl.mat',{'weights':weights})