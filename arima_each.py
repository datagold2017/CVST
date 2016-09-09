import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm
import util



directory = 'top/04-QEWDE0080DES.csv'
TIME = 3*15  # 20s * 3 * 15
ONE = 24*60*3/TIME

def selectDay(array, day=0):
    """
    :param array: the array shape like 8960*1
    :param day: 0 to 6, means Monday to Sunday
    :return: the selected sequence for each day and tick out the atypical days
    """
    res = []
    RES


    return res
array = np.genfromtxt(directory, delimiter=',')
array = util.nan2zero(array)
newa = util.divide_time(array, TIME)
period_array = np.reshape(newa,[newa.shape[0]*newa.shape[1],1])

start = datetime.datetime(2015,1,1,0,0,0)
delta = datetime.timedelta(seconds=TIME*20)
date_list = [start + delta*x for x in range(len(period_array))]

df = pd.DataFrame(period_array)
df['index'] = date_list
df.set_index(['index'], inplace=True)
df.index.name=None

df.columns=['traffic']
df['traffic'] = df.traffic.apply(lambda x: int(x))
df.traffic.plot(figsize=(120,8), title= '15min', fontsize=14)
# plt.show()

"""
decomposition = sm.tsa.seasonal_decompose(period_array,freq=96)
fig = plt.figure()
fig = decomposition.plot()
fig.set_size_inches(15, 8)
"""
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

test_stationarity(df.traffic) # not stationary
print "original finshed"
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
# test with first difference
df['first_difference'] = df.traffic - df.traffic.shift(1)
test_stationarity(df.first_difference.dropna(inplace=False))
print "first difference finished"

# seasonal first difference
df['seasonal_first_difference'] = df.first_difference - df.first_difference.shift(96)
test_stationarity(df.seasonal_first_difference.dropna(inplace=False))
print "seasonal first difference finished"

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.seasonal_first_difference.iloc[97:], lags=96, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.seasonal_first_difference.iloc[97:], lags=96, ax=ax2)
plt.show()
print "acf and pacf finished"

mod = sm.tsa.statespace.SARIMAX(df.traffic, trend='n', order=(1,0,0), seasonal_order=(0,1,1,96), \
                                mle_regression=True)
results = mod.fit()
print results.summary()
results.save("traffic_90days_15mins_1_0_0_0_1_1_96.pickle")

from statsmodels.iolib.smpickle import load_pickle
results = load_pickle("traffic_90days_15mins_1_0_0_0_1_0_96.pickle")

df['forecast'] = results.predict(start = 8000, end= 8500, dynamic= True)
df[['traffic', 'forecast']].ix[-1000:].plot(figsize=(12, 8))
plt.savefig('ts_df_predict_part.png', bbox_inches='tight')
