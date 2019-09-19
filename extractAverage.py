# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:41:50 2018

@author: conte
"""
import numpy as np


def extractAverage(Gtfilename, IntX, step):
    gt_File = open(Gtfilename, "rb")
    data = np.loadtxt(gt_File, delimiter=",", skiprows=0)

    timestamps = data[:, 0] + 1
    bpms = data[:, 1]

    IntBpm = []
    cont = 0
    endTime = 0

    while endTime <= timestamps[-1]:
        startTime = cont * step
        endTime = startTime + IntX

        if startTime > bpms.size:
            break

        if endTime > bpms.size:
            endTime = bpms.size

        cont += 1
        bpm = bpms[startTime:endTime]
        tempavg = np.average(bpm)
        IntBpm.append(tempavg)

    gt_File.close()

    return IntBpm
