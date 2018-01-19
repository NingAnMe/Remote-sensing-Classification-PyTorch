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


def get_ymd_and_hm(file_name):
    """
    从文件名中获取日期和时间
    :param file_name: (str)文件名
    :return:
    """
    pat = r'.*(\d{8})_(\d{4})'
    m = re.match(pat, file_name)
    ymd = m.group(1)
    hm = m.group(2)

    return ymd, hm


def is_cross_time(s_ymdhms1, e_ymdhms1, s_ymdhms2, e_ymdhms2):
    """
    判断俩个时间段是否有交叉
    :param s_ymdhms1: (int)第一个时间范围的开始时间
    :param e_ymdhms1: (int)第一个时间范围的结束时间
    :param s_ymdhms2: (int)第二个时间范围的开始时间
    :param e_ymdhms2: (int)第二个时间范围的结束时间
    :return: 布尔值
    """
    if s_ymdhms2 <= s_ymdhms1 <= e_ymdhms2:
        return True
    elif s_ymdhms2 <= e_ymdhms1 <= e_ymdhms2:
        return True
    elif s_ymdhms2 >= s_ymdhms1 and e_ymdhms2 <= e_ymdhms1:
        return True
    else:
        return False


def date_str2int(date_range):
    """
    将字符串格式的时间范围转换为整数，输出一个列表
    :param date_range: (str) YYYYMMDD-YYYYMMDD 或者 YYYYMM-YYYYMM
    :return: (list)
    """
    d = date_range.split('-')
    date_range = [int(i) for i in d]
    return date_range


def get_date_range(date_range):
    """
    将字符串格式的时间范围转换为 datetime
    :param date_range: (str) YYYYMMDD-YYYYMMDD 或者 YYYYMM-YYYYMM
    :return: (list)存放开始日期和结束日期
    """
    start_date, end_date = arg_str2date(date_range)
    return [start_date, end_date]


