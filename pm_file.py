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


def get_file_list(dir_path, pattern=''):
    """
    查找目录下的所有符合匹配模式的文件的绝对路径，包括文件夹中的文件
    :param dir_path: (str)目录路径
    :param pattern: (str)匹配模式 'hdf'
    :return: (list) 一个绝对路径列表，一个文件名列表
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
    获取文件的路径和文件名
    :param file_path: 文件的完整路径名
    :return:
    """
    if os.path.isfile(file_path):
        path, file_name = os.path.split(file_path)
        return [path, file_name]
    else:
        raise ValueError('文件不存在')


def filter_file_by_date(file_list, date_range):
    """
    过滤日期范围内的文件
    :param file_list: (list) 存放文件名的列表
    :param date_range: (str) YYYYMMDD-YYYYMMDD 或者 YYYYMM-YYYYMM
    :return:
    """
    new_file_list = []
    start_date, end_date = date_str2int(date_range)
    for file_name in file_list:
        ymd, hm = get_ymd_and_hm(file_name)
        file_date = int(ymd)
        if is_cross_time(start_date, end_date, file_date, file_date):
            new_file_list.append(file_name)
    return new_file_list


def filter_file_list(file_list, pattern='.*'):
    """
    过滤日期范围内的文件
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