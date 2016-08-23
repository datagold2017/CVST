# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 00:42:24 2016

@author: shenb
"""

import csv
import os
import numpy as np
import pickle


def checkFormat():
    """
    Each file should be a 90*4320 matrix (90 days, 4320 data points per day)
    :return: True or False list
    """
    directory = 'data/'
    all_csv = os.listdir(directory)
    res = []
    file = open('stats/format.txt', 'wb')
    for i, fcsv in enumerate(all_csv):
        if i == 10:
            break
        print i
        try:
            array = np.genfromtxt(directory + fcsv, delimiter=',')
            if array.shape[0] == 4320 and array.shape[1] == 90:
                res.append([fcsv, array.shape[0], array.shape[1], 1])
            else:
                res.append([fcsv, array.shape[0], array.shape[1], 0])
        except:
            res.append([fcsv, array.shape[0], array.shape[1], 0])

    pickle.dump(res,file)
    file.close()

def countNotPossible():
    f2 = open("stats/format.txt", "rb")
    load_list = pickle.load(f2)
    f2.close()
    count = 0
    for (k,v) in load_list.items():
        if v == 0:
            count += 1
            print k
    print count
    return

def isNaN(num):
    """
    To judge a number in the numpy getfromtxt method is a number or not
    :param num:
    :return: True if not nan
    """
    return num != num


def getNan(array):
    """
    get # of nan in a [N,1] array
    :param array: array with shape [1, N]
    :return: # of nan in a array
    """
    len = array.shape[1]
    res = np.zeros((array.shape[1], 1))
    for i in range(len):
        for j in range(array.shape[0]):
            if isNaN(array[j][i]):
                res[i][0] += 1

    return res

def getColNan():
    """
    get # of nan values in each column
    :return:
    """
    directory = 'data/'
    all_csv = os.listdir(directory)
    res = []

    for i, fcsv in enumerate(all_csv):
        # if i == 1:
        #     break
        print i
        array = np.genfromtxt(directory + fcsv, delimiter=',')
#        print array
        cnan = getNan(array)
        # print cnan
        # print np.sum(array)
        num_cnan = np.sum(cnan)
        print num_cnan
        res.append((fcsv, cnan, num_cnan))

    return res

def nan2zero(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if isNaN(array[i][j]):
                array[i][j] = 0
    return array


"""
Ranking all the files and calculate average traffic of each sensor

"""

# transfer empty and 'X' values (nan)into 0
# calculate each day with a number of nan values limit < 500/4030


def check():
    return


if __name__ == '__main__':
    # getColNan()
    checkFormat()
    # countNotPossible()