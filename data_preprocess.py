#-*- coding:utf-8 -*-
# @Time : 2019/9/28
# @Author : Botao Fan
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import os

import elo.data_exploration as de
import elo.about_time as abt


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

def drop_outlier(data, var, k):
    (scope_max, scope_min) = de.univar_outlier_scope(data, var, k)
    data.drop(
        index=data[(data[var] > scope_max) | (data[var] < scope_min)].index,
        inplace=True)


def process_train(data):
    data = data.copy()
    if 'target' in data.columns:
        drop_outlier(data, 'target', 3)
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
    #feature describe
    data['feature_max'] = data[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    data['feature_min'] = data[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    data['feature_mean'] = data[['feature_1', 'feature_2', 'feature_3']].mean(axis=1)
    data['feature_median'] = data[['feature_1', 'feature_2', 'feature_3']].median(axis=1)
    data['feature_std'] = data[['feature_1', 'feature_2', 'feature_3']].std(axis=1)
    return data


def get_train_test_values(train, test, y_name='target', include_cols=None):
    if y_name in test.columns and np.sum(~(test[y_name].isnull())) > 0:
        raise KeyError('Test dataframe should not contains %s' % y_name)
    all_data = train.append(test)
    if include_cols is None:
        col_names = all_data.columns
    else:
        col_names = include_cols
    train_x_vals = all_data.loc[~(all_data[y_name].isnull()), col_names].values
    test_x_vals = all_data.loc[(all_data[y_name].isnull()), col_names].values
    train_y_vals = all_data.loc[~(all_data[y_name].isnull()), y_name].values
    return train_x_vals, train_y_vals, test_x_vals, col_names


def transaction_to_feature(data):
    data = data.copy()
    #Deal with outliers
    print('Data has been copied!')
    data['installments'].replace(-1, np.nan, inplace=True)
    data['installments'].replace(999, np.nan, inplace=True)
    data.loc[data['purchase_amount'] > 0.8, 'purchase_amount'] = 0.8
    data.loc[data['category_2'].isnull(), 'category_2'] = 126
    data['category_2'] = data['category_2'].astype(np.float16)
    data.loc[data['category_2'] == 126, 'category_2'] = np.nan
    data['authorized_flag'] = data['authorized_flag'].map({'Y': 1, 'N': 0}).astype(np.int8)
    data['category_1'] = data['category_1'].map({'Y': 1, 'N': 0}).astype(np.int8)
    data['category_3'] = data['category_3'].map({'A': 0, 'B': 1, 'C': 2})
    data['price'] = data['purchase_amount'] / (data['installments'] + 0.0)
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    data['month'] = data['purchase_date'].dt.month
    data['day'] = data['purchase_date'].dt.day
    data['hour'] = data['purchase_date'].dt.hour
    data['weekofyear'] = data['purchase_date'].dt.weekofyear
    data['weekday'] = data['purchase_date'].dt.weekday
    data['weekend'] = (data['purchase_date'].dt.weekday >= 5).astype(np.int8)
    data['Christmas_Day_2017'] = abt.before_someday(data, 'purchase_date', '2017-12-25', 99)
    data['Mothers_Day_2017'] = abt.before_someday(data, 'purchase_date', '2017-06-04', 99)
    data['Fathers_Day_2017'] = abt.before_someday(data, 'purchase_date', '2017-08-13', 99)
    data['Children_day_2017'] = abt.before_someday(data, 'purchase_date', '2017-10-12', 99)
    data['Valentine_Day_2017'] = abt.before_someday(data, 'purchase_date', '2017-06-12', 99)
    data['Black_Friday_2017'] = abt.before_someday(data, 'purchase_date', '2017-11-24', 99)
    data['Mothers_Day_2018'] = abt.before_someday(data, 'purchase_date', '2018-05-13', 99)
    data['month_diff'] = (datetime.today() - data['purchase_date']).dt.days // 30
    data['month_diff'] = data['month_diff'] + data['month_lag']
    data['duration'] = data['purchase_amount']*data['month_diff']
    data['amount_month_ratio'] = data['purchase_amount']/(data['month_diff'] + 0.0)
    data = reduce_memory_usage(data)
    print('Data has been preprocess!')
    aggs = {}
    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']
    for col in col_unique:
        aggs[col] = ['nunique']
    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']
    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var', 'skew']
    aggs['installments'] = ['sum', 'max', 'mean', 'var', 'skew']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['month_diff'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['authorized_flag'] = ['mean']
    aggs['weekend'] = ['mean']#overwrite
    aggs['weekday'] = ['mean']#overwrite
    aggs['day'] = ['nunique', 'mean', 'min'] # overwrite
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size', 'count']
    aggs['price'] = ['sum', 'mean', 'max', 'min', 'var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['Fathers_Day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration'] = ['mean', 'min', 'max', 'var', 'skew']
    aggs['amount_month_ratio'] = ['mean', 'min', 'max', 'var', 'skew']
    features = data.groupby('card_id').agg(aggs)
    features.columns = pd.Index([e[0] + '_' + e[1] for e in features.columns.to_list()])
    features = reduce_memory_usage(features)
    return features


def get_train_valid_test_data(all_train, all_test, exclude_columns=[], valid_proportion=0.1):
    include_columns = [c for c in all_train.columns if c not in exclude_columns]
    train_valid_x, train_valid_y, test_x, column_names = \
        get_train_test_values(all_train, all_test, 'target', include_cols=include_columns)
    train_proportion = 1 - valid_proportion
    train_x, valid_x = train_valid_x[:int(len(train_valid_x) * train_proportion), :],  \
                       train_valid_x[int(len(train_valid_x) * train_proportion):, :]
    train_y, valid_y = train_valid_y[:int(len(train_valid_y) * train_proportion)],  \
                       train_valid_y[int(len(train_valid_y) * train_proportion):]
    return train_x, train_y, valid_x, valid_y, test_x, column_names



def train(all_train, all_test):
    include_cols = [c for c in all_train.columns if
                    c not in ['card_id', 'first_active_month', 'hist_purchase_date_max', 'hist_purchase_date_min',
                              'new_purchase_date_max', 'new_purchase_date_min']]
    train_x_values, train_y_values, test_x_values, col_names = \
        get_train_test_values(all_train, all_test, 'target', include_cols=include_cols)
    train_x, valid_x = \
        train_x_values[:int(len(train_x_values) * 0.9), :],  train_x_values[int(len(train_x_values) * 0.9):, :]
    train_y, valid_y = \
        train_y_values[:int(len(train_y_values) * 0.9)],  train_y_values[int(len(train_y_values) * 0.9):]
    lgb_train = lgb.Dataset(train_x, label=train_y)
    lgb_valid = lgb.Dataset(valid_x, label=valid_y)
    params = {
        'task': 'train',
        'boosting': 'goss',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'subsample': 0.9855232997390695,
        'max_depth': 7,
        'top_rate': 0.9064148448434349,
        'num_leaves': 63,
        'min_child_weight': 41.9612869171337,
        'other_rate': 0.0721768246018207,
        'reg_alpha': 9.677537745007898,
        'colsample_bytree': 0.5665320670155495,
        'min_split_gain': 9.820197773625843,
        'reg_lambda': 8.2532317400459,
        'min_data_in_leaf': 21,
        'verbose': -1,
    }
    reg = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        num_boost_round=2000,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    test_pred = reg.predict(test_x_values, num_iteration=reg.best_iteration)
    result = all_test[['card_id']]
    result['target'] = test_pred
    result.set_index('card_id', inplace=True)


if __name__ == '__main__':
    path = os.getcwd() + '/../data'
    raw_data = read_dir_csv(path)
    print('--------- Raw data has been read ---------')
    train, test = raw_data['train'], raw_data['test']
    train, test = reduce_memory_usage(train), reduce_memory_usage(test)
    train, test = process_train(train), process_train(test)
    print('--------- Preprocessing has been done ---------')
    tran_feats_hist = transaction_to_feature(raw_data['historical_transactions'])
    tran_feats_new = transaction_to_feature(raw_data['new_merchant_transactions'])
    print('--------- Features generating has benn done ---------')
    tran_feats_hist.columns = ['hist_' + c for c in tran_feats_hist.columns]
    tran_feats_new.columns = ['new_' + c for c in tran_feats_new.columns]
    train.set_index('card_id', inplace=True), test.set_index('card_id', inplace=True)
    all_train = train.join(tran_feats_hist).join(tran_feats_new)
    all_test = test.join(tran_feats_hist).join(tran_feats_new)
    all_train[all_train == np.inf] = np.nan
    all_train[all_train == -np.inf] = np.nan
    all_test[all_test == np.inf] = np.nan
    all_test[all_test == -np.inf] = np.nan
    all_train.to_csv(path + '/all_train.csv')
    all_test.to_csv(path + '/all_test.csv')



