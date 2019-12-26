#-*- coding:utf-8 -*-
# @Time : 2019/12/10
# @Author : Botao Fan
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import xgboost as xgb
from my_tools import timer


def cross_validate_hyperparams(train_x, train_y, estimator, hyper_param, cv=6, scoring='neg_mean_squared_error'):
    if len(train_x) != len(train_y):
        raise TypeError('Length of train_x should be same with train_y: %.0f, %.0f')
    gscv = GridSearchCV(estimator=estimator, param_grid=hyper_param, scoring=scoring, cv=cv, n_jobs=4, verbose=1)
    print "=========== Result ============"
    with timer('Training'):
        gscv.fit(train_x, train_y)
    print "Best Score is %.4f" % gscv.best_score_
    print "Best params:"
    print gscv.best_params_
    return gscv


if __name__=='__main__':
    param = {'subsample': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3,
              'min_child_weight': 1, 'n_estimators': 100, 'n_jobs': 4, 'random_state': 0, 'reg_alpha': 0,
              'reg_lambda': 1, 'scale_pos_weight': 1, 'silent': True}
    cv_param = {'n_estimators': [50, 75, 100, 150, 200]}
    xgb_estimator = xgb.XGBRegressor(**param)
    gscv = cross_validate_hyperparams(train_x, train_y, xgb_estimator, cv_param)


