# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:34:32 2018

@author: conte
"""
import sys

from utility.evaluateAlgorithm import evaluate

if len(sys.argv) != 5:
    print('Usage : evaluateAlg <gtfile> <algfile> <X range (in sec)> <X overlap (in sec)>')
    exit()

try:
    RMSE = evaluate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    print("RMSE: %.2f" % RMSE)

except Exception as error:
    print('Error: ')
    print(error)
