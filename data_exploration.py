#-*- coding:utf-8 -*-
# @Time : 2019/12/11
# @Author : Botao Fan
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot


def get_col_type(data):
    if type(data) is not pd.core.frame.DataFrame:
        raise TypeError('Input data is not dataframe but %s' % (type(data)))
    col_type = data.dtypes
    col_type.sort_values(inplace=True)
    col_type.columns = ['type']
    dtype_dict = dict()
    for t in col_type.unique():
        dtype_dict[str(t)] = list(col_type[col_type == t].index)
    # print '------ Types of All Data ------'
    # print col_type
    print '------ Number of Each Type ------'
    print col_type.groupby(col_type).count()
    return dtype_dict


def numerical_univar_plot(data, var_name):
    fig = plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    sns.distplot(data[var_name], fit=norm)
    plt.title(var_name)
    plt.subplot(3, 1, 2)
    probplot(data[var_name], plot=plt)
    plt.title('Skewness:%0.3f, Kurtosis:%0.3f' % (data[var_name].skew(), data[var_name].kurt()))
    plt.subplot(3, 1, 3)
    sns.boxplot(data[var_name])
    fig.tight_layout()
    plt.show()


def numerical_vars_plot(data, vars_list):
    for var in vars_list:
        if np.issubdtype(data[var].dtype, np.number):
            numerical_univar_plot(data, var)


def na_data_univar(data, var_name):
    missing_num = np.sum(data[var_name].isnull())
    num = len(data[var_name])
    return missing_num, missing_num / (num + 0.0)


def in_cols(data, cols):
    for c in cols:
        if c not in data.columns:
            raise KeyError("Column is not one of DataFrame's columns : %s" % c)
    return None


def get_cols(data, vars_list):
    if vars_list is None:
        cols = data.columns
    else:
        cols = vars_list
        in_cols(data, cols)
    return cols


def na_data_vars(data, vars_list=None):
    cols = get_cols(data, vars_list)
    data_num = data.shape[0]
    missing_num = data[cols].isnull().sum().sort_values(ascending=False).to_frame('count')
    missing_num['percent'] = missing_num['count'] / (data_num + 0.0)
    return missing_num


def vars_corr(data, vars_list=None, is_plot=True):
    cols = get_cols(data, vars_list)
    data_corr = data[cols].corr()
    if is_plot:
        sns.heatmap(data_corr, square=True)
        plt.show()
    return data_corr


def scatter_single(data, x, y):
    in_cols(data, [x, y])
    plt.scatter(data[x], data[y])
    plt.show()


def univar_outlier_scope(data, var, scope_k=1.5):
    quantile_75 = data[var].quantile(q=0.75)
    quantile_25 = data[var].quantile(q=0.25)
    iqr = quantile_75 - quantile_25
    scope_max = quantile_75 + scope_k * iqr
    scope_min = quantile_25 - scope_k * iqr
    return (scope_max, scope_min)


def category_univar_hist(data, var):
    plt.hist(data[var])
    plt.show()


def category_univer_y_box(data, var, y):
    sns.boxplot(data[var], data[y])
    plt.show()


def category_vars_hist(data, vars_list=None, plot_matrix=[2, 2]):
    cols = get_cols(data, vars_list)
    in_cols(data, cols)
    num = len(cols)
    if num > plot_matrix[0] * plot_matrix[1]:
        plot_matrix[1] = 3
        plot_matrix[0] = num / plot_matrix[1] + 1
    for i in range(num):
        plt.subplot(plot_matrix[0], plot_matrix[1], i + 1)
        plt.hist(data[vars_list[i]])
        plt.title(vars_list[i])
    plt.show()





