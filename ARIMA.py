import matplotlib.pylab as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import datetime
import matplotlib.dates as dates
import numpy as np
import os
import util
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }

weekname = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def getX(i, array):
    res = []
    j = i
    while j < len(array):
        res.append(array[j])
        j += 7
    return res


def getY(i, array):
    res = []
    j = i
   #  print 'array shape', array.shape[0]
    while j < array.shape[0]:
        res.append([array[j][0]])
        j += 7
    return res

def plotThreeMonths():
    directory = 'top/'
    all_csv = os.listdir(directory)
    fig = plt.figure(figsize=(30, 10))
    for i, fcsv in enumerate(all_csv):
        # if i == 2:
        #     break
        print i, fcsv
        array = np.genfromtxt(directory + fcsv, delimiter=',')
        array = util.nan2zero(array)
        newa = util.divide_time(array, 3*60*24)  # 3*20 = 60s = 1min

        x = np.arange(newa.shape[0]*newa.shape[1]) + 1
        newnewa = np.reshape(newa,[newa.shape[0]*newa.shape[1],1])

        plt.subplot(1,1,1)
        plt.title('Traffic on Three Months', fontdict=font)


        # plt.plot(x, newnewa)
        rect = plt.bar(x, newnewa, 0.8, alpha=0.4, color='g')
        plt.plot()

        for i in range(7):
            newx = getX(i, x)
           # print newx
            newy = getY(i, newnewa)
          #  print newy
            plt.plot(newx, newy, label=weekname[i])

        plt.xlabel('time (min)', fontdict=font)
        plt.ylabel('traffic', fontdict=font)
        plt.xticks(np.arange(90)+1)
        plt.legend(loc='lower right')
#        plt.show()
        plt.savefig('/home/byshen/CVST/pics/' + 'three' + fcsv  + '.png', dpi=100)
        plt.cla()
        plt.clf()
    return


def plotFirstMonthEachWeekday():
    directory = 'top/'
    all_csv = os.listdir(directory)
    fig = plt.figure(figsize=(16, 10))
    for i, fcsv in enumerate(all_csv):
        print i, fcsv

        for day in range(7):

            array = np.genfromtxt(directory + fcsv, delimiter=',')

            # print getNan(array)
            array = util.nan2zero(array)

            newa = util.divide_time(array, 180) # 180*20 = 3600s = 1h

            x = np.arange(newa.shape[0]) + 1

            plt.subplot(2, 4, day + 1)

            # plt.subplot(1,1,1)
            plt.title('Traffic on ' + weekname[day], fontdict=font)
            if day == 0:
                plt.plot(x, newa[:, day + 28], 's-', color="blue")
            else:
                plt.plot(x, newa[:, day + 0], 's-', color="blue")

            plt.plot(x, newa[:, day + 7], '*-', color="red")
            plt.plot(x, newa[:, day + 14], 'd-', color="green")
            plt.plot(x, newa[:, day + 21], '>-', color="black")
            plt.xlabel('time (h)', fontdict=font)
            # plt.ylabel('traffic', fontdict=font)
            if day == 0:
                plt.legend((str(day + 0 + 29), str(day + 7 + 1), str(day + 14 + 1), str(day + 21 + 1)),
                           loc='upper right', prop={'size': 10})
            else:
                plt.legend((str(day + 0 + 1), str(day + 7 + 1), str(day + 14 + 1), str(day + 21 + 1)),
                           loc='upper right', prop={'size': 10})

            # plt.show()

            plt.savefig('/home/byshen/CVST/pics/' + fcsv + '_all' + '.png', dpi=100)
        plt.cla()
        plt.clf()
    plt.close(fig)

if __name__ == '__main__':
    plotThreeMonths()