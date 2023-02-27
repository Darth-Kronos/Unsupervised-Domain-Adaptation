# Office-31
Office_31_root_path = r'/content/'
domain_a = 'amazon_amazon.csv'
domain_ad = 'amazon_dslr.csv'
domain_aw = 'amazon_webcam.csv'
domain_d = 'dslr_dslr.csv'
domain_da = 'dslr_amazon.csv'
domain_dw = 'dslr_webcam.csv'
domain_w = 'webcam_webcam.csv'
domain_wa = 'webcam_amazon.csv'
domain_wd = 'webcam_dslr.csv'
Office_31_name = ['A-D', 'A-W', 'D-A', 'D-W', 'W-A', 'W-D']

import numpy as np
import os
import torch.utils.data as data
from torchvision import datasets, transforms
import torch
from sklearn.cluster import SpectralClustering, KMeans
import collections
from scipy.io import loadmat
# import normalization_data


def separate_data(feas, labels, index, in_index=True):
    # 根据索引分离数据
    if in_index:
        feas = np.asarray([feas[i] for i in index])
        labels = np.asarray([labels[i] for i in index])

    else:
        feas = np.asarray([feas[i] for i in range(len(feas)) if i not in index])
        labels = np.asarray([labels[i] for i in range(len(labels)) if i not in index])
    return feas, labels


def concatenate_data(feas, labels, feas_extra, labels_extra):
    if len(np.asarray(feas).shape) == len(np.asarray(feas_extra).shape):
        feas = np.concatenate((feas, feas_extra), 0)
        labels = np.concatenate((labels, labels_extra), 0)
    return  feas, labels


def list_numpy(feas_list, labels_list):
    # 保存的时候是list，使用的时候就转为numpy
    feas = feas_list[0]
    labels = labels_list[0]
    for i in range(1, len(feas_list)):
        feas = np.concatenate((feas, feas_list[i]), 0)
        labels = np.concatenate((labels, labels_list[i]), 0)
    return feas, labels


def get_feas_labels(root_path, domain, fea_type='Resnet50'):
    # 得到原始特征
    path = os.path.join(root_path, domain)
    if fea_type == 'Resnet50':
        with open(path, encoding='utf-8') as f:
            imgs_data = np.loadtxt(f, delimiter=",")
            features = imgs_data[:, :-1]
            labels = imgs_data[:, -1]

    elif fea_type == 'MDS':
        # dict_keys(['__header__', '__version__', '__globals__', 'fts', 'labels'])
        domain_data = loadmat(path)
        features = np.asarray(domain_data['fts'])
        labels = np.asarray(domain_data['labels']).squeeze()

    else: # DeCAF6
        domain_data = loadmat(path)
        features = np.asarray(domain_data['feas'])
        labels = np.asarray(domain_data['labels']).squeeze() - 1  # start from 0
    return features, labels


def get_src_dataloader_by_feas_labels(feas, labels, batch_size=128, drop_last=False,
                                      normalization=False, fea_type='Resnet50'):
    # get dataloader
    if normalization:
        dataset = normalization_data.myDataset(feas, labels, fea_type)
    else:
        dataset = data.TensorDataset(torch.tensor(feas), torch.tensor(labels))
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    return dataloader


def get_ds_dtl_dtu(feas_ds, labels_ds, feas_dt, labels_dt, n_dtl):
    extra_index = np.random.choice(len(feas_dt), int(n_dtl * len(feas_dt)), replace=False)

    fea_tgt_label = np.asarray([feas_dt[i] for i in extra_index])
    labels_tgt_label = np.asarray([labels_dt[i] for i in extra_index])

    feas_dt = np.asarray([feas_dt[i] for i in range(len(feas_dt)) if i not in extra_index])
    labels_dt = np.asarray([labels_dt[i] for i in range(len(labels_dt)) if i not in extra_index])

    feas_ds = np.concatenate((feas_ds, fea_tgt_label), 0)
    labels_ds = np.concatenate((labels_ds, labels_tgt_label), 0)

    return feas_ds, labels_ds, feas_dt, labels_dt


def get_sd_td_with_labels_dataloader(root_path, ds, dt, n_Dtl, fea_type, batch_size=100):
    feas_src, labels_src = get_feas_labels(root_path, ds, fea_type=fea_type)
    feas_tgt, labels_tgt = get_feas_labels(root_path, dt, fea_type=fea_type)

    feas_src, labels_src, feas_tgt, labels_tgt = get_ds_dtl_dtu(feas_src, labels_src, feas_tgt, labels_tgt, n_Dtl)

    fea_type = data.TensorDataset(torch.tensor(feas_src), torch.tensor(labels_src))
    dataloader_src = data.DataLoader(
        dataset=fea_type,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    fea_type = data.TensorDataset(torch.tensor(feas_tgt), torch.tensor(labels_tgt))
    dataloader_tgt = data.DataLoader(
        dataset=fea_type,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    return dataloader_src, dataloader_tgt

import torch.utils.data as data
from torchvision import datasets, transforms
import os
import numpy as np


def get_path(root_path, images_domain):
    # 多个类别总结在一起，返回一个list
    images_path = []
    labels = []
    domain_classes_path = os.path.join(os.path.join(root_path, images_domain))
    domain_classes = os.listdir(domain_classes_path)
    domain_classes.sort(key=lambda x: int(x))

    for c in domain_classes:
        # 获取每一个类别的的路径。
        imgs_path = os.path.join(domain_classes_path, c)
        # 获取每一张图片的路径
        for img in os.listdir(imgs_path):
            img_path = os.path.join(domain_classes_path, c, img)
            images_path.append(img_path)
            labels.append(int(c))
    return images_path, labels


class myDataset(data.Dataset):
    def __init__(self, imgs_data, labels, feature_type):
        self.imgs_data = imgs_data
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5],[0.5])])
        if feature_type == 'DeCAF6':
            self.feature = 4096
        elif feature_type == 'Resnet50':
            self.feature = 2048
        elif feature_type == 'MDS':
            self.feature = 400

    def __getitem__(self, index):
        imgs_data = np.asarray(self.imgs_data[index])
        # imgs_data = imgs_data[:, np.newaxis]
        # 这里不知道为什么不加多一维，会报错，说输入只能是2或者3维
        # 加多一维 却变成[1,800,1] 太奇怪了，只能reshape
        # imgs_data = self.transform(Image.fromarray(imgs_data[:, np.newaxis])).reshape(1, 4096)
        imgs_data = self.transform(imgs_data[:, np.newaxis]).reshape(1, self.feature)
        return imgs_data, self.labels[index]

    # 可以直接打开路径
    # def __getitem__(self, index):
    #     img_data = self.transform(Image.open(self.imgs_path[index]))
    #     if img_data.shape[0] == 1:
    #         img_data = img_data.expand(3, img_data.shape[1], img_data.shape[2])
    #     return img_data, self.labels[index]

    def __len__(self):
        return len(self.imgs_data)
    
def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))


class OptimWithSheduler:
    # decay learning rate
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1


class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims  # if isinstance(optims, Iterable) else [optims]

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True
