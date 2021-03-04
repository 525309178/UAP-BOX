# coding:utf8
import os
import torch
import torch.nn as nn
from collections import namedtuple
from .vgg import VGG

###############################################################################
# Helper Functions  参考于论文 Perceptual Losses for Real-Time Style Transfer and Super-Resolution
###############################################################################
model_path = os.path.abspath(os.path.join(os.getcwd(), "..","Classfiers","CIFAR10","VGG_cifar10.t7"))
print(model_path)
#model_path = 'The path of pre-trained classfier， such as **/UAP-TOOL/Classfiers/CIFAR10/GoogLeNet_cifar10.t7'

class LossNetwork(torch.nn.Module):
    ''' 输入原始图像，输出中间层的特征图''' 
    def __init__(self):

        model = VGG('VGG19')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])

        super(LossNetwork, self).__init__()
        features = list(model.features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)


