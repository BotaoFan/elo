#-*- coding:utf-8 -*-
# @Time : 2019/12/17
# @Author : Botao Fan
from datetime import datetime
import pandas as pd
import numpy as np


def before_someday(data, var, someday='2017-12-25', max_before_day=99):
    return (pd.to_datetime(someday) - data[var]).dt.days.apply(
        lambda x: x if x > 0 and x <= max_before_day else 0)