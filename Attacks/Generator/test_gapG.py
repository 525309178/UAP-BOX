

from Generator import gapG
import numpy as np
from torch.autograd import Variable
import torch


center_crop =32
batchSize =4

# 定义模型
netG = gapG.ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', gpu_ids=[1])
netG.apply(gapG.weights_init)


# 定义噪声
noise_data = np.random.normal(0, 255, 32 * 32 * 3)

im_noise = np.reshape(noise_data, (3, center_crop, center_crop))
im_noise = im_noise[np.newaxis, :, :, :]
im_noise_tr = np.tile(im_noise, (batchSize, 1, 1, 1))
noise_tensor_tr = torch.from_numpy(im_noise_tr).type(torch.FloatTensor)
noise_tr = Variable(noise_tensor_tr).cuda(1)
print(noise_tr.size())

delta_im = netG(noise_tr)

print(delta_im.size())
print(delta_im[0])


noise = torch.FloatTensor(batchSize, 3, 32, 32)

noise = Variable(noise)
noise.resize_(batchSize, 3, 32, 32).normal_(0, 0.5)
delta_im1=netG(noise)
print(delta_im1.size())




