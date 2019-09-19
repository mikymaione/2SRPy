# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:34:32 2018

@author: conte
"""
import math

from extractAverage import extractAverage


def evaluate(gtfile, algfile, x_range, x_overlap):
    IntAlg = []
    IntGT = extractAverage(gtfile, int(x_range), int(x_overlap))

    algfile = open(algfile, 'r')

    for x in algfile:
        try:
            IntAlg.append(float(x))
        except Exception as error:
            print('Error: ')
            print(error)

    algfile.close()

    a = len(IntAlg)
    b = len(IntGT)

    if a > b:
        l = b
    else:
        l = a

    RMSE = 0.0

    for c in range(l):
        RMSE = RMSE + (IntAlg[c] - IntGT[c]) ** 2

    RMSE = RMSE / len(IntAlg)
    RMSE = math.sqrt(RMSE)

    if not (0 <= RMSE <= 100):
        raise Exception("RMSE not in range!")

    return RMSE
