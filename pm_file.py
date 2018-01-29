# coding:utf-8
"""
pm_file.py
目录和文件相关函数
~~~~~~~~~~~~~~~~~~~
creation time : 2018 1 19
author : anning
~~~~~~~~~~~~~~~~~~~
"""

import os
import sys
import logging
import re
from datetime import datetime, timedelta
from posixpath import join

import h5py
import numpy as np
from configobj import ConfigObj

from pm_time import get_ymd_and_hm, is_cross_time, get_date_range, str2date


def get_file_list(dir_path, pattern=''):
    """
    查找目录下的所有符合匹配模式的文件的绝对路径，包括文件夹中的文件
    :param dir_path: (str)目录路径
    :param pattern: (str)匹配模式 'hdf'
    :return: (list) 绝对路径列表
    """
    file_list = []
    # 递归查找目录下所有文件
    for root, dir_list, file_names in os.walk(dir_path):
        for i in file_names:
            if pattern in i:
                file_list.append(os.path.join(root, i))
    return file_list


def get_path_and_name(file_path):
    """
    通过一个绝对地址获取文件的所在文件夹和文件名
    :param file_path: (str)文件的完整路径名
    :return: (list)[路径, 文件名]
    """
    if os.path.isfile(file_path):
        path, file_name = os.path.split(file_path)
        return [path, file_name]
    else:
        raise ValueError('value error: not a file_path')


def filter_file_list(file_list, pattern='.*'):
    """
    过滤符合匹配模式的文件
    :param file_list: (list) 存放文件名的列表
    :param pattern: (str) 匹配规则
    :return:
    """
    new_file_list = []
    for file_name in file_list:
        m = re.match(pattern, file_name)
        if m:
            new_file_list.append(file_name)
    return new_file_list


def filter_dir_by_date_range(dir_path, start_date, end_date):
    """
    过滤日期范围内的目录
    :return:
    """
    dirs = os.listdir(dir_path)
    tem_dir_list = []
    for dir_name in dirs:
        dir_date_start = str2date(dir_name)
        dir_date_end = dir_date_start + timedelta(days=365) - timedelta(seconds=1)
        if is_cross_time(dir_date_start, dir_date_end, start_date, end_date):
            tem_dir_list.append(os.path.join(dir_path, dir_name))
    return tem_dir_list
