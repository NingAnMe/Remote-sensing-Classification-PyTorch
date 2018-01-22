# coding:utf-8
"""
pm_h5py.py
hdf5 处理相关函数
~~~~~~~~~~~~~~~~~~~
creation time : 2018 1 19
author : anning
email : anning@kingtansin.com
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


def read_hdf5_dataset(file_path, set_name):
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


def compress_hdf5(pre_hdf5, out_dir=None, out_file=None, level=5):
    """
    对 hdf5 文件进行压缩
    :param pre_hdf5: 输入文件
    :param out_dir: 输出路径，文件名与原来相同
    :param out_file: 输出文件，使用输出文件名
    :param level: 压缩等级
    :return:
    """
    if not os.path.isfile(pre_hdf5):
        raise ValueError('is not a file')
    if out_dir is not None:
        path, name = os.path.split(pre_hdf5)
        new_hdf5 = os.path.join(out_dir, name)
    elif out_file is not None:
        new_hdf5 = out_file
    else:
        raise ValueError('outpath and outfile value is error')
    pre_hdf5 = h5py.File(pre_hdf5, 'r')
    new_hdf5 = h5py.File(new_hdf5, 'w')

    compress(pre_hdf5, new_hdf5)

    pre_hdf5.close()
    new_hdf5.close()


def compress(pre_object, out_object, level=5):
    """
    对 h5df 文件进行深复制，同时对数据表进行压缩
    :param pre_object: 
    :param out_object: 
    :return: 
    """
    for key in pre_object.keys():
        pre_dateset = pre_object.get(key)

        if type(pre_dateset).__name__ == "Group":
            out_dateset = out_object.create_group(key)
            compress(pre_dateset, out_dateset)
        else:
            out_dateset = out_object.create_dataset(key, dtype=pre_dateset.dtype, data=pre_dateset,
                                                    compression='gzip', compression_opts=level,  # 压缩等级5
                                                    shuffle=True)
            # 复制dataset属性
            for akey in pre_dateset.attrs.keys():
                out_dateset.attrs[akey] = pre_dateset.attrs[akey]

    # 复制group属性
    for akey in pre_object.attrs.keys():
        out_object.attrs[akey] = pre_object.attrs[akey]
