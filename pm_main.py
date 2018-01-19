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


def get_config(file_path, file_name):
    """
    获取配置信息
    :param file_path: (str)配置文件目录
    :param file_name: (str)文件名
    :return: (configobj)
    """
    config_file = join(file_path, file_name)
    if os.path.isfile(config_file):
        config_obj = ConfigObj(config_file)
        return config_obj
    else:
        raise ValueError('配置文件不存在')


def get_main_dir():
    """
    获取主程序所在目录
    :return: (str)
    """
    main_dir_path, main_file = os.path.split(os.path.realpath(__file__))
    return main_dir_path
