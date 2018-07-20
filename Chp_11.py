from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, date
from datetime import timedelta
from dateutil.parser import parse
import pytz
from scipy.stats import percentileofscore
pd.options.display.width = 500

path1 = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\examples\stock_px_2.csv'
path2 = r'C:\Users\Marlombrei\Downloads\deliverable-bonds-conversion-factors.xls'

df = pd.read_excel(path2)
print(df)
test1 = datetime.now()
test2 = date.today()


# now = datetime.now()
# print(now,'\n')
# 
# print(now.year, now.month, now.day,'\n')
# 
# delta = datetime(2011,1,7) - datetime(2008,6,24,8,15)
# print(delta,'\n')
# print(delta.days,'\n')
# print(delta.seconds,'\n')
# 
# start = datetime(2011,1,7)
# end = datetime(2008,6,24,8,15)
# 
# print(start + timedelta(12),'\n')
# print(start + 2*timedelta(14),'\n')
# 
# print(start,'\n')
# print(str(start),'\n')
# print(start.strftime('%Y-%m-%d'),'\n')
# 
# value = '2011-01-03'
# print(datetime.strptime(value,'%Y-%m-%d'),'\n')
# 
# datestrs = ['7/6/2011','8/6/2011']
# 
# print([datetime.strptime(dt,'%m/%d/%Y') for dt in datestrs],'\n')
# 
# print([parse('2011-01-03')],'\n')
# 
# dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
#          datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
# 
# ts = pd.Series(np.random.randn(6), index=dates)
# print(ts,'\n')
# 
# print(ts + ts[::2],'\n')
# 
# print(ts.index.dtype,'\n')
# 
# stamp = ts.index[2]
# print([stamp],'\n')
# 
# print(ts[stamp],'\n')
# 
# print(ts['1/10/2011'],'\n')
# 
# print(ts['20110110'],'\n')
# 
# longer_ts = pd.Series(np.random.randn(1000),
#                       index=pd.date_range(start='1/1/2000', periods=1000))
# 
# #print(longer_ts[:20])
# 
# print(longer_ts['2001-05'],'\n')
# 
# print(ts[datetime(2011,1,7):],'\n')
# print(ts,'\n')
# print(ts['1/6/2011':'1/11/2011'],'\n')
# print(ts['2011'],'\n')
# 
# print(ts.truncate(after='1/9/2011'),'\n')
# 
# dates = pd.date_range(start='1/1/2000',periods=100, freq='W-WED')
# print(dates,'\n')
# long_df = pd.DataFrame(np.random.randn(100,4), index=dates,
#                        columns=['Colorado','Texas','New York','Ohio'])
# print(long_df.loc['5-2001'],'\n')
# 
# 
# dates = pd.DatetimeIndex(['1/1/2000','1/2/2000','1/2/2000','1/2/2000','1/3/2000'])
# dup_ts = pd.Series(np.arange(5), index=dates)
# print(dup_ts,'\n')
# print(dup_ts.index.is_unique,'\n')
# 
# print(dup_ts.groupby(level=0).count(),'\n')
# 
# 
# print(pytz.common_timezones[:10],'\n')
# 
# tz = pytz.timezone('America/New_York')
# print([tz],'\n')
# 
# rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts,'\n')
# print(ts.index.tz,'\n')
# 
# print(pd.date_range('3/9/12 09:30',periods=10, freq='D', tz='UTC'),'\n')
# 
# print(ts,'\n')
# 
# ts_utc = ts.tz_localize(tz='UTC')
# print('ts_utc','\n',ts_utc,'\n')
# print(ts_utc.index,'\n')
# 
# print(ts_utc.tz_convert('America/New_York'),'\n')
# 
# ts_eastern = ts.tz_localize('America/New_York')
# print(ts_eastern.tz_convert('UTC'),'\n')
# print('Europe/Berlin','\n',ts_eastern.tz_convert('Europe/Berlin'),'\b')
# 
# print(ts.index.tz_localize('Asia/Shanghai'))

# stamp = pd.Timestamp('2011-03-12 04:00')
# print([stamp],'\n')
# 
# stamp_utc = stamp.tz_localize('utc')
# print([stamp_utc],'\n')
# print([stamp_utc.tz_convert('America/New_York')],'\n')
# 
# print(stamp_utc.value,'\n')
# print(stamp_utc.tz_convert('America/New_York').value,'\n')
# 
# from pandas.tseries.offsets import Hour
# 
# stamp = pd.Timestamp('2012-03-12 01:30', tz='US/Eastern')
# print([stamp],'\n')
# print([stamp + Hour()],'\n')
# 
# stamp2 = pd.Timestamp('2012-11-04 00:30', tz='US/Eastern')
# print([stamp2],'\n')
# print([stamp2 + 2 * Hour()],'\n')

# data = pd.read_csv(path)
# print(data.head(5),'\n')
# 
# index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
# print(index,'\n')
# data.index = index
# print(data.head(),'\n')
# 
# rng = pd.date_range('2000-01-01',periods=100, freq='D')
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts.head(),'\n')
# 
# #===============================================================================
# # resample(rule, how, axis, fill_method, closed, label, convention, kind, loffset, limit, base, on, level)
# #===============================================================================
# 
# print(ts.resample(rule='M', kind='period').mean())

# rng = pd.date_range('2000-01-01', periods=12, freq='T')
# ts = pd.Series(np.arange(len(rng)), index=rng)
# print(ts,'\n')
# 
# print (ts.resample(rule='5min', closed='right').sum(),'\n')
# 
# print(ts.resample('5min').ohlc())

# frame = pd.DataFrame(np.random.randn(2,4),
#                      index= pd.date_range('1/1/2018', periods=2, freq='W-WED'),
#                      columns=['Colorado','Texas','New York','Ohio'])
# 
# print(frame,'\n')
# 
# df_daily = frame.resample(rule='W-THU').ffill(limit=2)
# print(df_daily,'\n')
# 
# frame = pd.DataFrame(np.random.randn(24,4),
#                      index= pd.period_range('1-2000', periods=24, freq='M'),
#                      columns=['Colorado','Texas','New York','Ohio'])
# print(frame,'\n')
# 
# annual_frame = frame.resample('A-DEC').mean()
# print(annual_frame,'\n')
# 
# print(annual_frame.resample('Q-DEC').ffill(),'\n')
# print(annual_frame.resample('Q-DEC', convention='end').ffill(),'\n')



# close_px_all = pd.read_csv(path, parse_dates=True, index_col=0)
# close_px = close_px_all[['AAPL','MSFT','XOM']]
# #print(close_px[:20],'\n')
# close_px = close_px.resample('B').ffill()#, how, axis, fill_method, closed, label, convention, kind, loffset, limit, base, on, level)
# #print(close_px[:20],'\n')
# appl_std250 = close_px.AAPL.rolling(255, min_periods=20).std()
# #print(appl_std250[:20])
# expanding_mean = appl_std250.expanding().mean()
#print(expanding_mean[:30])
#plt.plot(close_px.AAPL)
#close_px.AAPL.rolling(250).mean().plot()
#expanding_mean.plot()
#appl_std250.plot()
#close_px.rolling(60).mean().plot(logy=True)

# aapl_px = close_px.AAPL['2006':'2011']
# #print(aapl_px)
# close_px.AAPL.plot()
# ma60 = aapl_px.rolling(60, min_periods=20).mean()#simple moving average
# ma60.plot(label='Simple MA')
# ewma60 = aapl_px.ewm(span=60).mean()
# ewma60.plot(label='EWMA')
# plt.legend()
# plt.show()

# spx_px = close_px_all['SPX']
# spx_rets = spx_px.pct_change()
# 
# returns = close_px.pct_change()
# #corr = returns.AAPL.rolling(125, min_periods=100).corr(spx_rets)
# corr = returns.rolling(125, min_periods=100).corr(spx_rets)
# #corr.plot()
# #plt.show()
# 
# 
# 
# score_at_2pct = lambda x: percentileofscore(x,0.02)
# 
# result = returns.AAPL.rolling(250).apply(score_at_2pct)
# result.plot()
# plt.show()




































































































































