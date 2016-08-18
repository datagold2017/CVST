# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 15:10:45 2016

@author: shenb
"""

import csv
import os

# print os.getcwd()\
directory = 'J:\\2015_1'
all_csv = os.listdir(directory+'\\occ')

if not os.path.exists(directory + '\\new'):
    os.mkdir(directory+'\\new')
    
print len(all_csv)

for i, fcsv in enumerate(all_csv):
   # if i == 1:
  #      break
    print i
    reader = csv.reader(file(directory + '\\occ\\' + fcsv, 'rb'))
    writer = csv.writer(file(directory + '\\new\\'+ fcsv, 'wb'))
    for index, line in enumerate(reader):
        #print line[1:]
        if index > 4:
            writer.writerow(line[1:len(line) - 1])

