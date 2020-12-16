# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import os
import argparse
import random
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning.metrics.functional as plm
from ranger.ranger2020 import Ranger

from flyai.dataset import Dataset
from flyai.utils.log_helper import train_log

from model import Model
from net import get_net
from path import MODEL_PATH, DATA_PATH
import numpy as np

'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=5, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=3, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
'''
dataset.get_step() 获取数据的总迭代次数
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

images_train, labels_train, images_val, labels_val = dataset.get_all_data()


# print(len(x_train))
# print(x_train)
# print(y_train)
# exit()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


############
# 辅助函数
############


def f1_score(preds, target):
    preds = torch.argmax(preds, dim=1)
    return plm.f1(preds, target, num_classes=Num_classes)


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


############
# 常量
############
Epoch = args.EPOCHS
Batch = args.BATCH
Suffix = ''
Loss_val_min = np.inf
Num_classes = 45
Net_name = 'resnext101_32x8d'  # resnet50  resnext101_32x8d
Optimizer_name = 'Adam'  # Adam  Ranger
Lr_scheduler_name = None  # None  ReduceLROnPlateau
'''
定义模型
'''
# 网络
net = get_net(Net_name).to(device)

acc_function = f1_score

# # 学习率调整器
# if Lr_scheduler_name == 'ReduceLROnPlateau':
#     lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2,
#                                                               verbose=True)
#     scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=lr_scheduler)
# else:
#     lr_scheduler = None

############
# 日志
############
log_dir = './log/{}-{}-{}-{}-{}-{}'.format(Net_name, Optimizer_name,
                                           Lr_scheduler_name, Epoch, Batch, Suffix)
logger = SummaryWriter(log_dir=log_dir)

loss_train = 0
loss_val = 0
acc_train = 0
acc_val = 0


class Classifier(LightningModule):
    # def __init__(self, net, loss_function='CrossEntropyLoss', optim='Adam', lr=1e-3):
    #     super(Classifier, self).__init__()
    #     self._net = net
    #     self._loss_function = self.configure_loss_function(loss_function)
    #     self._optim = optim
    #     self._lr = lr

    def __init__(self):
        super(Classifier, self).__init__()
        self._net = net
        self._loss_function = self.configure_loss_function('CrossEntropyLoss')
        self._optim = 'Adam'
        self._lr = 1e-3

    def forward(self, x):
        return self._net(x)

    @staticmethod
    def configure_loss_function(loss_function):
        # 损失函数
        if loss_function == 'CrossEntropyLoss':
            lf = torch.nn.CrossEntropyLoss()
        elif loss_function == 'LabelSmoothCrossEntropyLoss':
            lf = LabelSmoothCrossEntropyLoss(smoothing=0.1)
        else:
            raise ValueError(loss_function)
        return lf

    def configure_optimizers(self):
        if self._optim == 'Adam':
            optim = torch.optim.Adam(self.parameters(), lr=self._lr)
        elif self._optim == 'Ranger':
            optim = Ranger(net.parameters(), lr=self._lr)
        else:
            raise ValueError(self._optim)
        return optim

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self._loss_function(pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self._loss_function(pred, y)
        return loss

    def test_step(self, batch, batch_idx):
        """
        model.eval() and torch.no_grad() are called automatically for testing.
        The test loop will not be used until you call: trainer.test()
        .test() loads the best checkpoint automatically
        """
        x, y = batch
        pred = self.forward(x)
        loss = self._loss_function(pred, y)
        return loss


class ClassifierDataset(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images, labels, transformI=None, transformM=None):
        super(ClassifierDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transformI = transformI
        self.transformM = transformM

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
                #  torchvision.transforms.Resize((128,128)),
                # torchvision.transforms.CenterCrop(96),
                # torchvision.transforms.RandomRotation((-10, 10)),
                # torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            # self.lx = torchvision.transforms.Compose([
            #     #  torchvision.transforms.Resize((128,128)),
            #     # torchvision.transforms.CenterCrop(96),
            #     # torchvision.transforms.RandomRotation((-10, 10)),
            #     # torchvision.transforms.Grayscale(),
            #     torchvision.transforms.ToTensor(),
            #     # torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            # ])
            self.lx = torchvision.transforms.Compose([
                np.array,
                torch.from_numpy,
            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        label_list = ['apron', 'bare-land', 'baseball-field', 'basketball-court', 'beach', 'bridge', 'cemetery',
                      'church', 'commercial-area', 'desert',
                      'dry-field', 'forest', 'golf-course', 'greenhouse', 'helipad', 'ice-land', 'island', 'lake',
                      'meadow', 'mine',
                      'mountain', 'oil-field', 'paddy-field', 'park', 'parking-lot', 'port', 'railway',
                      'residential-area', 'river', 'road',
                      'roadside-parking-lot', 'rock-land', 'roundabout', 'runway', 'soccer-field', 'solar-power-plant',
                      'sparse-shrub-land', 'storage-tank', 'swimming-pool', 'tennis-court',
                      'terraced-field', 'train-station', 'viaduct', 'wind-turbine', 'works']
        image = Image.open(os.path.join(DATA_PATH, self.images[idx]['img_path']))
        label = label_list.index(self.labels[idx]['label'])

        # apply this seed to img tranfsorms
        seed = np.random.randint(0, 100)  # make a seed with numpy generator
        random.seed(seed)
        torch.manual_seed(seed)
        x = self.tx(image)
        y = label

        return x, y


dataset_train = ClassifierDataset(images_train, labels_train)
dataset_val = ClassifierDataset(images_val, labels_val)

train_loader = DataLoader(dataset_train)
val_loader = DataLoader(dataset_val)

trainer = pl.Trainer()
model = Classifier()
Trainer.fit(model, dataset_train, dataset_val)

# for step in range(dataset.get_step()):
#
#     print('Step: {}/{}'.format(step + 1, dataset.get_step()))
#
#     if lr_scheduler:
#         lr_scheduler.step(step)
#
#     net.train()
#
#     x_train, y_train = dataset.next_train_batch()
#     x_val, y_val = dataset.next_validation_batch()
#
#     x_train = torch.Tensor(x_train).to(device)
#     y_train = torch.Tensor(y_train).long().to(device)
#
#     # train
#     optimizer.zero_grad()
#     pred_train = net(x_train)
#     loss_train = loss_function(pred_train, y_train)
#     loss_train.backward()
#     optimizer.step()
#     loss_train = loss_train.item()
#
#     # acc_train
#     y_train = y_train.detach()
#     pred_train = pred_train.detach()
#     acc_train = acc_function(pred_train, y_train)
#     acc_train = acc_train.item()
#
#     with torch.no_grad():
#         net.eval()
#         # loss_val
#         loss_val = 0
#         pred_val = None
#
#         y_val = torch.Tensor(y_val).long().to(device)
#         num_val = x_val.shape[0]
#         for i in range(0, num_val, Batch):
#             _x = x_val[i:i + Batch]
#             _x = torch.Tensor(_x).to(device)
#             _y = y_val[i:i + Batch]
#             _pred_val = net(_x)
#             loss_val += loss_function(_pred_val, _y).item() / (num_val / Batch)
#             if pred_val is None:
#                 pred_val = _pred_val
#             else:
#                 pred_val = torch.cat((pred_val, _pred_val), dim=0)
#
#         # acc_val
#         y_val = y_val.detach()
#         pred_val = pred_val.detach()
#         acc_val = acc_function(pred_val, y_val)
#         acc_val = acc_val.item()
#
#     '''
#     实现自己的模型保存逻辑
#     '''
#     if loss_val < Loss_val_min:
#         model.save_model(net, MODEL_PATH, overwrite=True)
#         Loss_val_min = loss_val
#
#     logger.add_scalars('Loss', {
#         'Train Loss': loss_train,
#         'Val Loss': loss_val,
#     }, global_step=step + 1)
#     logger.add_scalars('Acc', {
#         'Train Acc': acc_train,
#         'Val Acc': acc_val,
#     }, global_step=step + 1)
#
#     train_log(train_loss=loss_train, train_acc=acc_train, val_acc=acc_val, val_loss=loss_val)
#
# print('loss_val_min : {}'.format(Loss_val_min))
