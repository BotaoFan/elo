#-*- coding:utf-8 -*-
# @Time : 2019/12/10
# @Author : Botao Fan
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from my_tools import timer

def cross_validate_hyperparams(train_x, train_y, estimator, hyper_params, cv=6, scoring='neg_mean_squared_error'):
    if len(train_x) != len(train_y):
        raise TypeError('Length of train_x should be same with train_y: %.0f, %.0f')
    gscv = GridSearchCV(estimator=estimator, param_grid=hyper_params, scoring=scoring, n_jobs=4, verbose=1)
    print "=========== Result ============"
    with timer('Training'):
        gscv.fit(train_x, train_y)
    print "Best Score is %.4f" % np.sqrt(gscv.best_score_)
    print "Best params:"
    print gscv.best_params_
    return gscv