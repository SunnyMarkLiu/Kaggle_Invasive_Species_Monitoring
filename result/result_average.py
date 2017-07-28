#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-28 下午4:26
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from conf.configure import Configure

result_files = os.listdir('./model_results/')

result_0_99243 = pd.read_csv('model_results/inception_v3_0.99243.csv')
result_0_99176 = pd.read_csv('model_results/vgg16_0.99176.csv')
result_0_99046 = pd.read_csv('model_results/vgg19_0.99046.csv')
result_0_99004 = pd.read_csv('model_results/resnet_50_0.99004.csv')

final_result = pd.DataFrame()
final_result['name'] = result_0_99243['name']

# average
final_result['invasive'] = 0.5 * (0.6 * result_0_99243['invasive'] + 0.4 * result_0_99176['invasive']) + \
                           0.5 * (0.6 * result_0_99046['invasive'] + 0.4 * result_0_99004['invasive'])

final_result.to_csv(Configure.submission_path.format('final_average_result'), index=False)
