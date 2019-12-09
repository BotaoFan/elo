#-*- coding:utf-8 -*-
# @Time : 2019/9/28
# @Author : Botao Fan
import os
import pandas as pd
import numpy as np
import tensorflow as tf

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)

def read_dir(data_path = ''):
    if not os.path.isdir(data_path):
        raise IOError('Directory does not exist: %s' % data_path)
    all_files = []
    raw_data = {}
    for root, dirs, files in os.walk(data_path):
        all_files += files
    for file in all_files:
        try:
            raw_data[file.replace('.csv', '')] = pd.read_csv(data_path + '/' + file)
        except:
            print('The file is not a csv file: %s' % file)
    return raw_data

if __name__ == '__main__':
    path = '/Users/botaofan/PycharmProjects/elo'
    data_path = path + '/data'
    raw_data = read_dir(data_path)