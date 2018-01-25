# coding:utf-8
"""
publicmodels_test.py
单元测试模块
~~~~~~~~~~~~~~~~~~~
creation time : 2018 1 25
author : anning
~~~~~~~~~~~~~~~~~~~
"""
import os
import unittest

from pm_main import get_config


class TestMain(unittest.TestCase):
    """
    pm_main 测试
    """
    def setUp(self):
        # 创建配置文件
        with open('test_config.cfg', 'w') as f:
            f.write('[PATH]\n')
            f.write('PATH = /root/home/user')

    def tearDown(self):
        # 删除配置文件
        os.remove('test_config.cfg')

    def test_get_config(self):

        config = get_config('.', 'test_config.cfg')
        self.assertEqual(config['PATH']['PATH'], '/root/home/user')
