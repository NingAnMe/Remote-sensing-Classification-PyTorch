# coding:utf-8
"""
pm_calculate.py
计算相关函数
~~~~~~~~~~~~~~~~~~~
creation time : 2018 1 19
author : anning
~~~~~~~~~~~~~~~~~~~
"""

import os
import sys
import logging
import re
from datetime import datetime
from posixpath import join

import h5py
import numpy as np
from configobj import ConfigObj


def get_avg_and_std(dataset, data_range):
    """
    计算平均值和标准差
    :param dataset: (np.ndarray)获取的数据集
    :param data_range: (int)截取范围大小
    :return:(list)
    """
    # 获取数据表的维数
    rank = dataset.ndim

    if rank == 2:  # 与通道数无关
        # 获取轴数
        shape = dataset.shape
        dim = int(shape[0])
        avg_and_std = calculate_avg_and_std(dataset, dim, data_range)
        return avg_and_std

    elif rank == 3:  # 多条通道
        # 获取轴数
        shape = dataset.shape
        channel_num = int(shape[0])  # 通道数
        dim = int(shape[1])  # 每个通道的数据轴数

        # 记录每条通道的均值和标准差
        channels_avg_and_std = []
        for i in xrange(0, channel_num):
            dataset_tem = dataset[i]
            avg_and_std = calculate_avg_and_std(dataset_tem, dim, data_range)
            channels_avg_and_std.append(avg_and_std)
        return channels_avg_and_std

    else:
        return ['-nan', '-nan']


def calculate_avg_and_std(dataset, dim, data_range=3):
    """
    计算均值和标准差
    :param dataset: 一个二维数据列表
    :param dim: 轴数
    :param data_range: 范围大小
    :return:
    """
    if len(dataset) != 0:
        # 获取切片的位置坐标
        num_start = int(dim / 2) - int(data_range / 2) - 1
        num_end = int(dim / 2) + int(data_range / 2)

        # 对数据表进行切片
        dataset = dataset[num_start:num_end, num_start:num_end]

        # 计算均值和标准差
        avg = np.mean(dataset)
        std = np.std(dataset)

        if avg == -999:
            return ['-nan', '-nan']
        else:
            return [avg, std]
