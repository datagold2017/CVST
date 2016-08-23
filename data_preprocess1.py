# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 15:10:45 2016

@author: shenb
"""

import csv
import os

"""
This is for first step to clean data, to modify data in a matrix format
so that simply use numpy to process it.
"""
def data2matrix():
    directory = '/home/byshen/CVST/2015_2'
    all_csv = os.listdir(directory+'/occ')

    if not os.path.exists(directory + '/new'):
        os.mkdir(directory+'/new')

    print len(all_csv)

    for i, fcsv in enumerate(all_csv):
        print i
        reader = csv.reader(file(directory + '/occ/' + fcsv, 'rb'))
        writer = csv.writer(file(directory + '/new/'+ fcsv, 'wb'))
        # print reader
        num_row = 0
        for index, line in enumerate(reader):
            num_row += 1
        print num_row
        reader = csv.reader(file(directory + '/occ/' + fcsv, 'rb'))
        for index, line in enumerate(reader):
            """
            if index > 4:
                # BUG existed here,
                # not all csv file has 4 lines of explanation
                # just choose the last 4320 line is fine
                writer.writerow(line[1:len(line) - 1])
            """
            if index > num_row - 4320 - 1:
                writer.writerow(line[1:len(line) - 1])

def read_data():
    directory = '/home/byshen/CVST/2015_2'
    all_csv = os.listdir(directory + '/new')

    for i, fcsv in enumerate(all_csv):
        if i == 1:
            break
        print i
        reader = csv.reader(file(directory + '/new/' + fcsv, 'rb'))
        for index, line in enumerate(reader):
            print line


data2matrix()
