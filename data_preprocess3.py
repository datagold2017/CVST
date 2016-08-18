# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 15:28:45 2016

@author: shenb
"""

import csv
import os

"""
This is for
 Pick the locations with three months data(use python dict)
 Combine all the three months data into one file
"""
def crwkjbc():
    d1 = {'a':1}
    ch = 'a'
    if ch not in d1:
        d1.update({ch:1})
    else:
        d1[ch] +=1
    print d1

def pick_locations():
    directory = '/home/byshen/CVST/'
    csv1 = os.listdir(directory + '2015_1/new')
    csv2 = os.listdir(directory + '2015_2/new')
    csv3 = os.listdir(directory + '2015_3/new')

    res = {}
    for i in csv1:
        if i not in res:
            res.update({i:1})
    for j in csv2:
        if j not in res:
            res.update({j:1})
        else:
            res[j] +=1
    for k in csv3:
        if k not in res:
            res.update({k:1})
        else:
            res[k] += 1

    tick_csv = []

    for key in res.keys():
        if res[key] != 3:
            tick_csv.append(key)
    for csv in tick_csv:
        print csv
    return tick_csv
"""
401DE0391DWE.csv
401DW0081DES.csv
401DE0010DWC.csv
401DE0451DEE.csv
401DE0391DWC.csv
401DE0010DWE.csv
401DE0431DEC.csv
401DE0010DWR.csv
"""


"""
Remove the not 3 files
move to abort folder
"""
def mv_files(files):
    directories=['/home/byshen/CVST/2015_1/new/',\
                 '/home/byshen/CVST/2015_2/new/',\
                 '/home/byshen/CVST/2015_3/new/']
    dest = '/home/byshen/CVST/abort_csv/'
    for directory in directories:
        for file in files:
            if os.path.exists(directory + file):
                bashcmd = 'cp %s%s %s_%s' %(directory, file, \
                                             directory, dest)
                print bashcmd
                os.system(bashcmd)
                bashcmd = 'rm %s/%s' % (directory, file)
                print bashcmd
                os.system(bashcmd)

mv_files(pick_locations())