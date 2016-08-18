# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:20:18 2016

@author: shenb
"""
import matplotlib  
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
# import csv
import os
import numpy as np

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
        
weekname =['Monday', 'Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
def isNaN(num):
    return num != num
    
def getNan(array):
    len = array.shape[1]
    res = np.zeros((array.shape[1],1))
    for i in range(len):
        for j in range(array.shape[0]):
            if isNaN(array[j][i]):
                res[i][0] += 1
                
    return res
         

def getColNan():
    directory = 'J:\\2015_1'
    all_csv = os.listdir(directory+'\\new')

    for i, fcsv in enumerate(all_csv):
        #if i == 1:
        #    break
        print i
        array = np.genfromtxt(directory + '\\new\\' + fcsv, delimiter=',')
        print array
        print getNan(array)


def nan2zero(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if isNaN(array[i][j]):
                array[i][j] = 0
    return array
    
def divide_time(array, N):
    len = array.shape[1]
    wid = array.shape[0]
    
    neww = array.shape[0]/N + int(array.shape[0]%N > 0)
    
    print neww, len
    new_array = np.zeros((neww, len))
    for j in range(len):
        for i in range(neww):
            for k in range(N):
                if i*N+k >= wid:
                    break
                new_array[i][j] += array[i*N+k][j]
                
    return new_array


directory = 'J:\\2015_1'
all_csv = os.listdir(directory+'\\new')    

def plot():
    fig = plt.figure(figsize=(16,10))  
    for i, fcsv in enumerate(all_csv):
        if i < 82:
            continue
        print i
        
        for day in range(7):
            
            array = np.genfromtxt(directory + '\\new\\' + fcsv, delimiter=',')
        
            #print getNan(array)
            array = nan2zero(array)        
            
            newa = divide_time(array, 180)
            
            x = np.arange(newa.shape[0])+1
            
            
            plt.subplot(2,4,day+1)
            
            #plt.subplot(1,1,1)
            plt.title('Traffic on ' + weekname[day], fontdict=font)
            if day == 0:
                plt.plot(x, newa[:,day + 28], 's-', color="blue")
            else:
                plt.plot(x, newa[:,day + 0], 's-', color="blue")
                
            plt.plot(x, newa[:,day + 7], '*-', color="red")
            plt.plot(x, newa[:,day + 14], 'd-', color="green")
            plt.plot(x, newa[:,day + 21], '>-', color="black")
            plt.xlabel('time (h)', fontdict=font)
            #plt.ylabel('traffic', fontdict=font)
            if day == 0:
                plt.legend((str(day+0+29), str(day+7+1), str(day+14+1),str(day+21+1)),
                       loc='upper right', prop={'size':10})
            else:
                plt.legend((str(day+0+1), str(day+7+1), str(day+14+1),str(day+21+1)),
                       loc='upper right', prop={'size':10})
                 
            # plt.show()
                   
            plt.savefig(directory + '\\pics\\'+fcsv +'_all' + '.png',dpi=100) 
        plt.cla()
        plt.clf()
    plt.close(fig)
    
plot()


