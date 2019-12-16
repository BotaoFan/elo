#-*- coding:utf-8 -*-
# @Time : 2019/9/28
# @Author : Botao Fan
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from datetime import datetime

import data_exploration as de

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)


def read_dir_csv(data_path=''):
    if not os.path.isdir(data_path):
        raise IOError('Directory does not exist: %s' % data_path)
    all_files = []
    raw_data = {}
    for root, dirs, files in os.walk(data_path):
        all_files += files
    for file in all_files:
        try:
            raw_data[file.replace('.csv', '')] = pd.read_csv(data_path + '/' + file)
            print('%s has been read' % file)
        except:
            print('Warning: The file is not a csv file: %s' % file)
    return raw_data


def one_hot_encoder(df, na_as_column=True, category_cols=None):
    if category_cols is None:
        category_cols = list(df.columns)
    target_cols = [col for col in category_cols if df[col].dtype == 'object']
    all_columns = list(df.columns)
    df = pd.get_dummies(df, columns=target_cols, dummy_na=na_as_column)
    new_columns = [col for col in df.columns if col not in all_columns]
    return df, new_columns


def reduce_memory_usage(df):
    memory_start = df.memory_usage().sum() / 1024.0**2
    numeric_type = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    type_max, type_min = dict(), dict()
    type_max['int8'], type_max['int16'], type_max['int32'], type_max['int64'] = \
        np.iinfo(np.int8).max, np.iinfo(np.int16).max, np.iinfo(np.int32).max, np.iinfo(np.int64).max
    type_min['int8'], type_min['int16'], type_min['int32'], type_min['int64'] = \
        np.iinfo(np.int8).min, np.iinfo(np.int16).min, np.iinfo(np.int32).min, np.iinfo(np.int64).min
    type_max['float16'], type_max['float32'], type_max['float64'] = \
        np.finfo(np.float16).max, np.finfo(np.float32).max, np.finfo(np.float64).max
    type_min['float16'], type_min['float32'], type_min['float64'] = \
        np.finfo(np.float16).min, np.finfo(np.float32).min, np.finfo(np.float64).min
    for col in df.columns:
        col_type = str(df[col].dtype)
        if col_type in numeric_type:
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
    memory_end = df.memory_usage().sum()/1024.0**2
    memory_decrease = memory_start - memory_end
    print('Memory usage after optimaztion is {:2f} MB while raw is {:2f} MB'.format(memory_end, memory_start) )
    print('Decreased by {:2f}'.format(memory_decrease))
    return df


def process_train(data):
    data = data.copy()
    if 'target' in data.columns:
        (scope_max, scope_min) = de.univar_outlier_scope(data, 'target', 3)
        data.drop(
            index=data[(data['target'] > scope_max) | (data['target'] < scope_min)].index,
            inplace=True)
    else:
        data['target'] = np.nan
    #date features
    data['first_active_month'] = pd.to_datetime(data['first_active_month'])
    data['quarter'] = data['first_active_month'].dt.quarter
    data['elapsed_time'] = (datetime.today() - data['first_active_month']).dt.days
    data['days_feature1'] = data['elapsed_time'] * data['feature_1']
    data['days_feature2'] = data['elapsed_time'] * data['feature_2']
    data['days_feature3'] = data['elapsed_time'] * data['feature_3']
    data['days_feature1_ratio'] = data['feature_1'] / (data['elapsed_time'] + 0.0)
    data['days_feature2_ratio'] = data['feature_2'] / (data['elapsed_time'] + 0.0)
    data['days_feature3_ratio'] = data['feature_3'] / (data['elapsed_time'] + 0.0)
    data, cols = one_hot_encoder(data, na_as_column=False)
    #feature describe
    data['feature_max'] = data[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    data['feature_min'] = data[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    data['feature_mean'] = data[['feature_1', 'feature_2', 'feature_3']].mean(axis=1)
    data['feature_median'] = data[['feature_1', 'feature_2', 'feature_3']].median(axis=1)
    data['feature_std'] = data[['feature_1', 'feature_2', 'feature_3']].std(axis=1)
    return data


def get_train_test_values(train, test, y_name = 'target', include_cols = None):
    if y_name in test.columns:
        raise KeyError('Test dataframe should not contains %s' % y_name)
    all_data = train.append(test)
    if include_cols is None:
        col_names = all_data.columns
    train_x_vals = all_data.loc[~(all_data[y_name].isnull()), col_names].values
    test_x_vals = all_data.loc[(all_data[y_name].isnull()), col_names].values
    train_y_vals = all_data.loc[~(all_data[y_name].isnull()), y_name].values
    return train_x_vals, train_y_vals, test_x_vals, col_names











if __name__ == '__main__':
    path = '/Users/botaofan/PycharmProjects/elo'
    data_path = path + '/data'
    raw_data = read_dir_csv(data_path)
    train, test = raw_data['train'], raw_data['test']
    train, test = reduce_memory_usage(train), reduce_memory_usage(test)
    train, test = process_train(train), process_train(test)

