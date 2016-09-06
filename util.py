import matplotlib.pyplot as plt
# import csv
import os
import numpy as np


def isNaN(num):
    return num != num


def getNan(array):
    len = array.shape[1]
    res = np.zeros((array.shape[1], 1))
    for i in range(len):
        for j in range(array.shape[0]):
            if isNaN(array[j][i]):
                res[i][0] += 1

    return res


def nan2zero(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if isNaN(array[i][j]):
                array[i][j] = 0
    return array


def divide_time(array, N):
    len = array.shape[1]
    wid = array.shape[0]

    neww = array.shape[0] / N + int(array.shape[0] % N > 0)

    print neww, len
    new_array = np.zeros((neww, len))
    for j in range(len):
        for i in range(neww):
            for k in range(N):
                if i * N + k >= wid:
                    break
                new_array[i][j] += array[i * N + k][j]

    return new_array