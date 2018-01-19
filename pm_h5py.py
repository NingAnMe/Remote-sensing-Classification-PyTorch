# coding:utf-8

import os
import sys
import logging
import re
from datetime import datetime
from posixpath import join

import h5py
import numpy as np
from configobj import ConfigObj


def read_hdf5(file_path, set_name):
    """
    读取 hdf5 文件，返回一个 numpy 多维数组
    :param file_path: (unicode)文件路径
    :param set_name: (str)表的名字
    :return: baseset：(numpy.ndarray)读取到的数据
    """
    dataset = []
    if os.path.isfile(file_path):
        file_h5py = h5py.File(file_path, 'r')
        data = file_h5py.get(set_name)[:]
        dataset = np.array(data)
        file_h5py.close()
        return dataset
    else:
        return dataset


