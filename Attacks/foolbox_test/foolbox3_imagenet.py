import argparse
import os
import time
import torch
import torchvision
import torchvision.models as models
import eagerpy as ep
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

from foolbox.attacks import FGSM
from foolbox.attacks import LinfDeepFoolAttack,LinfBasicIterativeAttack,EADAttack,LinfProjectedGradientDescentAttack
from foolbox import PyTorchModel, accuracy, samples

import sys
sys.path.append('../')
from Utils.utils import progress_bar


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--targeted', type=int, default=0, help='if the attack is targeted (default False)')
parser.add_argument('--chosen_target_class', type=int, default=0, help='int representing class to target')

parser.add_argument('--targetClassifier', default='incv3', help="incv3 or densenet or vgg16")
parser.add_argument('--imagenetVal', type=str, default='E:\\dataset\\ImageNet\\val\\val_sub', help='int representing class to target')
parser.add_argument('--outf', default='./LPGD_incv3_0.05', help='folder to output images and model checkpoints')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 1. load raw dataset.....define normalization means and stddevs
model_dimension = 299 if opt.targetClassifier == 'incv3' else 256
center_crop = 299 if opt.targetClassifier == 'incv3' else 224

mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

# 2. get source image and label
print('==> Preparing data..')
normalize = transforms.Normalize(mean=mean_arr,
                                 std=stddev_arr)

data_transform = transforms.Compose([
    transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor(),

])
rawset = torchvision.datasets.ImageFolder(root=opt.imagenetVal, transform=data_transform)
rawloader = DataLoader(dataset=rawset, num_workers=opt.workers, batch_size=opt.testBatchSize, shuffle=True)


# 3. instantiate a model
model = models.inception_v3(pretrained=True).eval()
# model = models.densenet121(pretrained=True).eval()
# model = models.vgg16(pretrained=True).eval()
model = model.cuda()

preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)



# 4. define attack param

# attack = FGSM()
# attack = EADAttack()
attack = LinfProjectedGradientDescentAttack()
# attack = LinfDeepFoolAttack()
# attack = LinfBasicIterativeAttack()
#TODO 下方的最大值就

# 是约束的最大扰动范数阈值，255像素下的10就是1像素下的0.04
epsilons = [
    0.049
       ]
nepochs = 0
rawacc = 0.0
advacc = 0.0
if __name__ == "__main__":
    total = 0
    attack_success = 0
    start = time.time()
    print("==> Testing %s.." %(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))))

    # 4 make adversarial sample
    for batch_idx, (images, labels) in enumerate(rawloader):
        total += labels.size(0)
        nepochs = nepochs + 1

        if batch_idx > 10000:
            break
        images, labels = ep.astensors(images.cuda(),labels.cuda())

        # TODO 注：转换类型，否则损失函数报错
        labels = labels.astype(dtype=torch.long)

        # TODO 计算对抗样本
        advs, _, success = attack(fmodel, images, labels, epsilons=epsilons)
        # calculate and report the robust accuracy
        robust_accuracy = 1 - success.float32().mean(axis=-1)



        rawacc += accuracy(fmodel,images,labels)

        for adv,label,acc in zip(advs,labels,robust_accuracy):
            if not os.path.exists(os.path.join(opt.outf, str(label.item()))):
                os.mkdir(os.path.join(opt.outf, str(label.item())))
            advacc = advacc + acc.item()*100.0
            torchvision.utils.save_image(adv.raw[0], '{}/{}/_{}.png'.format(opt.outf, label, batch_idx, ))  # 保存3D 0-1 tensor

            progress_bar(batch_idx, len(rawloader),
                         "Raw success rate: %.2f, Robust_accuracy: %.2f,linf_norm: %.2f" % (100.0*rawacc/total
                                                                                      ,advacc/total,
                                                                                      (adv - images).norms.linf(axis=(1, 2, 3)).numpy()
                             ))
    end = time.time()
    print("==> Test end %s.." %(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))))