#-*- coding:utf-8 -*-
# @Time : 2019/12/10
# @Author : Botao Fan

import datetime
import pandas as pd
import numpy as np
import warnings
import time

from contextlib import contextmanager

@contextmanager
def timer(title):
    start_time = time.time()
    yield
    end_time = time.time()
    print('{} is Done in {:.0f}s'.format(title, end_time - start_time))

