import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm
import util
import time

from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=96)
    rolstd = pd.rolling_std(timeseries, window=96)

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput


TIME = 3*15  # 20s * 3 * 15
ONE = 24*60*3/TIME # 24*60*3/ 15*3


def selectDay(array, day=0):
    """
    :param array: the array shape like 8960*1
    :param day: 0 to 6, means Monday to Sunday
    :return: the selected sequence for each day and tick out the atypical days
    """
    res = []
    for i in range(13): # 13* 7 = 91
        realweekday = day + 7*i
        if day == 0:
            if realweekday in [0]: # holiday
                continue
        elif day == 1:
            if realweekday in [1]: # holiday
                continue
        elif day == 2:
            if realweekday in [2]: # holiday
                continue
        elif day == 3:
            if realweekday in [73]:# some break down during the morning
                continue
        elif day == 4:
            if realweekday in [36, 42]:
                continue
        elif day == 5:
            if realweekday in [12, 75]:
                continue

        print i, realweekday, len(res)
        res = np.append(res, array[(realweekday)*ONE : (realweekday+1)*ONE, 0])

    return res
def myreshape(array):
    """
    :param array: numpy.ndarray
    :return: to list , column by column, not by row!!!
    """
    res = []
    for i in range(array.shape[1]):
        res = np.append(res, array[:,i])
    res = np.reshape(res, [len(res), 1])
    return res

def arima_day(DAY):
    directory = 'top/04-QEWDE0080DES.csv'

    file_name = "1_0_0_0_1_1_%d_day%d" % (ONE, DAY)

    array = np.genfromtxt(directory, delimiter=',')
    array = util.nan2zero(array)
    newa = util.divide_time(array, TIME)
    period_array = myreshape(newa)

    oneperiod = selectDay(period_array, DAY)

    testperiod = oneperiod[-ONE:]
    oneperiod = oneperiod[:-ONE]

    start = datetime.datetime(2015,1,1,0,0,0)
    delta = datetime.timedelta(seconds=TIME*20)
    date_list = [start + delta*x for x in range(len(oneperiod))]

    df = pd.DataFrame(oneperiod)
    df['index'] = date_list
    df.set_index(['index'], inplace=True)
    df.index.name=None

    df.columns=['traffic']
    df['traffic'] = df.traffic.apply(lambda x: int(x))
    """
    df.traffic.plot(figsize=(120,8), title= '15min', fontsize=14)
    plt.show()
    """
    """
    decomposition = sm.tsa.seasonal_decompose(oneperiod,freq=ONE)
    fig = plt.figure()
    fig = decomposition.plot()
    fig.set_size_inches(15, 8)


    test_stationarity(df.traffic) # not stationary
    print "original finshed"
    """
    """
    # test with logarithmic
    df.traffic_log= df.traffic.apply(lambda x: np.log(x))
    test_stationarity(df.traffic_log)
    print "log finished"

    # test with first difference
    df['first_difference'] = df.traffic - df.traffic.shift(1)
    test_stationarity(df.first_difference.dropna(inplace=False))
    print "first difference finished"


    # test with log first difference
    df['log_first_difference'] = df.traffic_log - df.traffic_log.shift(1)
    test_stationarity(df.log_first_difference.dropna(inplace=False))
    print "log first difference finished"

    # seasonal first difference
    df['seasonal_first_difference'] = df.first_difference - df.first_difference.shift(12)
    test_stationarity(df.seasonal_first_difference.dropna(inplace=False))
    print "seasonal first difference finished"

    # log seasonal first difference
    df['log_seasonal_first_difference'] = df.log_first_difference - df.log_first_difference.shift(12)
    test_stationarity(df.log_seasonal_first_difference.dropna(inplace=False))
    print "log seasonal first difference"
    """

    """
    # test with first difference
    df['first_difference'] = df.traffic - df.traffic.shift(1)
    test_stationarity(df.first_difference.dropna(inplace=False))
    print "first difference finished"

    # seasonal first difference
    df['seasonal_first_difference'] = df.first_difference - df.first_difference.shift(ONE)
    test_stationarity(df.seasonal_first_difference.dropna(inplace=False))
    print "seasonal first difference finished"

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df.seasonal_first_difference.iloc[ONE+1:], lags=ONE, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df.seasonal_first_difference.iloc[ONE:], lags=ONE, ax=ax2)
    plt.show()
    print "acf and pacf finished"
    """

    mod = sm.tsa.statespace.SARIMAX(df.traffic, trend='n', order=(1,0,0), seasonal_order=(0,1,1,ONE), \
                                    mle_regression=True)
    results = mod.fit()
    print results.summary()
    results.save("%s.pickle" % file_name)
    from statsmodels.iolib.smpickle import load_pickle
    results = load_pickle("%s.pickle" % file_name)

    testdf = pd.DataFrame(testperiod)

    start = datetime.datetime(2015, 1, len(oneperiod)/ONE + 1, 0, 0, 0)
    delta = datetime.timedelta(seconds=TIME * 20)
    date_list = [start + delta * x for x in range(len(testperiod))]

    testdf['index'] = date_list
    testdf.set_index(['index'], inplace=True)
    testdf.index.name = None

    testdf.columns = ['true_value']
    testdf['true_value'] = testdf.true_value.apply(lambda x: int(x))

    # prediction = results.predict(start = len(oneperiod), end= len(oneperiod) + ONE, dynamic= True)

    testdf['forecast'] = results.predict(start = len(oneperiod), end= len(oneperiod) + ONE, dynamic= True)

    testdf[['true_value', 'forecast']].ix[:].plot(figsize=(12, 8))
    # testdf[['forecast']].ix[:].plot(figsize=(12, 8))
    print len(oneperiod)
    print len(testperiod)
    # print prediction
    # print "----------"
    # print testdf['forecast']
    # print "----------"
    # print testdf['true_value']
    # plt.show()
    plt.title(directory + file_name)
    plt.savefig('%s.png' %(file_name), bbox_inches='tight')
    testdf['relative_error'] = abs(testdf['forecast'] - testdf['true_value']) / testdf['true_value']
    testdf.to_csv('%s_prediction_res.csv' % file_name)
    return np.mean(testdf['relative_error'])


if __name__ == '__main__':
    start = time.time()
    result = 0
    for day in range(7):
       result += arima_day(day)
    print result/1

    end = time.time()
    print end - start