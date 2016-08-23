# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:02:38 2016

@author: shenb
"""
import pickle, csv, os
import numpy as np
import matplotlib.pyplot as plt


def lastElement(s):
    return s[-1]

def analyze():
    f2 = open("stats/analysis.txt", "rb")
    load_list = pickle.load(f2)
    f2.close()

    ranklist = sorted(load_list, key=lastElement, reverse=True)

    for i in range(100):
        print ranklist[i][0], '\t\t', np.average(ranklist[i][1]), '\t\t', \
            np.max(ranklist[i][1]), '\t\t', ranklist[i][3]


    # plot to analyse
    fig = plt.figure(figsize=(16, 10))

    x = np.arange(len(ranklist)) + 1
    y = [term[3]  for i,term in enumerate(ranklist)]
    plt.plot(x, y)
    plt.xlabel('rank')
    plt.ylabel('total traffic')
    plt.title("Total traffic with ranking")
    plt.show()


def chooseTopN(N):
    f2 = open("stats/analysis.txt", "rb")
    load_list = pickle.load(f2)
    f2.close()

    ranklist = sorted(load_list, key=lastElement, reverse=True)

    count = 0
    for i in range(200):
        print count
        if count == N:
            break
        if np.average(ranklist[i][1]) <= 200:
            # This file is WRONG, full of zeros and 100!!!
            if ranklist[i][0] == '401DE1200DWR.csv':
                continue

            bashcmd = "cp data/%s top/%02d-%s" % (ranklist[i][0], count, ranklist[i][0])
            print bashcmd
            os.system(bashcmd)
            count += 1
        """
        print ranklist[i][0], '\t\t', np.average(ranklist[i][1]), '\t\t', \
            np.max(ranklist[i][1]), '\t\t', ranklist[i][3]
        """
if __name__ == '__main__':
    chooseTopN(30)