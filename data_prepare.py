#-*- coding:utf-8 -*-
# @Time : 2019/9/28
# @Author : Botao Fan
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)


def read_dir(data_path=''):
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


def one_hot_encoder(df, na_as_column=True, category_cols=None):
    if category_cols is None:
        category_cols = list(df.columns)
    target_cols = [col for col in category_cols if df[col].dtype == 'object']
    all_columns = list(df.columns)
    df = pd.get_dummies(df, columns=target_cols, dummy_na=na_as_column)
    new_columns = [col for col in df.columns if col not in all_columns]
    return df, new_columns


def reduct_memory_usage(df):
    memory_start = df.memory_usage().sum() / 1024.0**2
    numeric_type = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    type_max, type_min = dict(), dict()
    type_max['int8'], type_max['int16'], type_max['int32'], type_max['int64'] = \
        np.iinfo(np.int8).max, np.iinfo(np.int16).max, np.iinfo(np.int32).max, np.iinfo(np.int64).max
    type_max['float16'], type_max['float32'], type_max['float64'] = \
        np.iinfo(np.float16).max, np.iinfo(np.float32).max, np.iinfo(np.float64).max
    for col in df.columns:
        col_type = str(df[col].dtype)
        if col in numeric_type:
            c_max = df[col].max()
            c_min = df[col].min()
            if col_type[:3] == 'int':
                if c_max < type_max['int8'] and c_min > type_min['int8']:
                    df[col] = df[col].astype(np.int8)
                elif c_max < type_max['int16'] and c_min > type_min['int16']:
                    df[col] = df[col].astype(np.int16)
                elif c_max < type_max['int32'] and c_min > type_min['int32']:
                    df[col] = df[col].astype(np.int32)
                elif c_max < type_max['int64'] and c_min > type_min['int64']:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_max < type_max['float16'] and c_min > type_min['float16']:
                    df[col] = df[col].astype(np.float16)
                elif c_max < type_max['float32'] and c_min > type_min['float32']:
                    df[col] = df[col].astype(np.float32)
                elif c_max < type_max['float64'] and c_min > type_min['float64']:
                    df[col] = df[col].astype(np.float64)
    memory_end = df.memory_usage.sum() / 1024.0**2
    memory_decrease = memory_start - memory_end
    print('Memory usage after optimaztion is {:2f} MB while raw is {:2f} MB'.format(memory_end, memory_start) )
    print('Decreased by {:2f}'.format(memory_decrease))
    return df







if __name__ == '__main__':
    path = '/Users/botaofan/PycharmProjects/elo'
    data_path = path + '/data'
    raw_data = read_dir(data_path)