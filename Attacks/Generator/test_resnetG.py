# Success ..............................................
from Generator import resnetG
import numpy as np
import torch
from torch.autograd import Variable

netG = resnetG.define_G(3, 3, 64, 'resnet_6blocks',
                          norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
print(netG)
center_crop = 32
batchSize = 4

noise_data = np.random.uniform(0, 255, center_crop * center_crop * 3)
im_noise = np.reshape(noise_data, (3, center_crop, center_crop))
im_noise = im_noise[np.newaxis, :, :, :]
im_noise_tr = np.tile(im_noise, (batchSize, 1, 1, 1))

noise_tensor_tr = torch.from_numpy(im_noise_tr).type(torch.FloatTensor)
noise_tr = Variable(noise_tensor_tr)
print(noise_tr.size())

result = netG(noise_tr)
print(result.size())