from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle




# 
# 
# frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
#                   index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
#                   columns=[['Ohio', 'Ohio', 'Colorado'],
#                            ['Green', 'Red', 'Green']])
# 
# print(frame,'\n')
# frame.index.names = ['key_1', 'key_2']
# frame.columns.names = ['State','Colour']
# print(frame,'\n')
# print(frame.swaplevel(0,1).sort_index(),'\n')
# print(frame.sum(),'\n')
# print(frame.sum(level=0),'\n')
# print(frame.sum(level='Colour', axis=1),'\n')

# frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
#                    'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
#                    'd': [0, 1, 2, 0, 1, 2, 3]})
# 
# print(frame,'\n')
# 
# frame2 = frame.set_index(['c','d'], drop=False)# drop, append, inplace, verify_integrity)
# print(frame2,'\n')
# 
# print(frame2.reset_index())


# df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
# df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3,6)})
#   
# print(df1,'\n')
# print(df2,'\n')
# 
# print(pd.merge(df1,df2, on='key'),'\n')

# df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
# df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
# print(df3,'\n')
# print(df4,'\n')
# 
# print(pd.merge(df3,df4, left_on='lkey', right_on='rkey', how='inner'),'\n')
# print(pd.merge(df3,df4, left_on='lkey', right_on='rkey', how='outer').fillna('Marlom'),'\n')

# df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
# df2 = pd.DataFrame({'key': ['a', 'b', 'a', 'b', 'd','b'], 'data2': range(6)})
# print(df1,'\n')
# print(df2,'\n')
# 
# print(pd.merge(df1,df2, on='key'),'\n')
# print(pd.merge(df1,df2, on='key', how='left'),'\n')
# print(pd.merge(df1,df2, on='key', how='inner'),'\n')

# left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'], 'key2': ['one', 'two', 'one'],'lval': [1, 2, 3]})
# right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'], 'key2': ['one', 'one', 'one', 'two'], 'rval': [4, 5, 6, 7]})
# print(left,'\n')
# print(right,'\n')
# 
# print(pd.merge(left, right, on='key1', suffixes=('_left', '_right'), indicator=True),'\n')
# 
# left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
# right1 = pd.DataFrame({'group_val': [3.5, 7,5]}, index=['a', 'b','a'])
# print(left1,'\n')
# print(right1,'\n')
# 
# print(pd.merge(left1, right1, left_on='key', right_index=True, how='outer'))

# lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
#                    'key2': [2000, 2001, 2002, 2001, 2002],
#                    'data': np.arange(5.)})
# 
# righth = pd.DataFrame(np.arange(12).reshape((6, 2)),
#                    index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'], [2001, 2000, 2000, 2000, 2001, 2002]],
#                    columns=['event1', 'event2'])
# 
# 
# print(lefth,'\n')
# print(righth,'\n')
# print(pd.merge(lefth,righth, left_on=['key1','key2'], right_index=True, how='outer'),'\n') #(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator)
# print(pd.merge(righth,lefth, left_index=True, right_on=['key1','key2'], how='outer'))

# left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'], columns=['Ohio', 'Nevada'])
# 
# right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]], index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
# 
# print(left2,'\n')
# print(right2,'\n')
# 
# print(pd.merge(left2, right2, how='outer', left_index=True, right_index=True, sort=True),'\n')
# print(left2.join(right2, how='outer'),'\n')

# a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=['f', 'e', 'd', 'c', 'b', 'a'])
# b = pd.Series(np.arange(len(a), dtype=np.float64), index=['f', 'e', 'd', 'c', 'b', 'a'])
# 
# print(a,'\n')
# print(b,'\n')
# 
# print(b.combine_first(a))
# print(a.combine_first(other=b))
# 
# 
# print(pd.Series.combine_first(b,a))

# df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan], 'b': [np.nan, 2., np.nan, 6.], 'c': range(2, 18, 4)})
# 
# df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.], 'b': [np.nan, 3., 4., 6., 8.]})
# print(df1,'\n')
# print(df2,'\n')
# print(df1.combine_first(df2))

# data = pd.DataFrame(np.arange(6).reshape((2, 3)),
#                  index=pd.Index(['Ohio', 'Colorado'], name='state'),
#                  columns=pd.Index(['one', 'two', 'three'], name='number'))
#  
# # print(data,'\n')
#  
# result = data.stack()
# print(result,'\n')
# # print(result.unstack(),'\n')
# # print(result.unstack(level='number'),'\n')
# 
# 
# s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
# s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
#  
# # print(s1,'\n')
# # print(s2,'\n')
#  
# data2 = pd.concat([s1,s2], keys=['one','two'])
# print(data2,'\n')
# print(data2.unstack(),'\n')
#  
# print(data2.unstack().stack(),'\n')
# print(data2.unstack().stack(dropna=False),'\n')

# df = pd.DataFrame({'left': result, 'right': result + 5}, columns=pd.Index(['left', 'right'], name='side'))
# 
# print(df,'\n')
# 
# print(df.unstack(level='state'),'\n')#(level, fill_value)
# 
# print(df.unstack().stack(level='side'),'\n')

# path = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\examples\macrodata.csv'
# data = pd.read_csv(path)
# 
# print(data.head(),'\n')
# 
# periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')
# columns = pd.Index(['realgdp','infl','unemp'], name='item')
# 
# data = data.reindex(columns=columns)
# data.index = periods.to_timestamp(freq='D', how='end')
# ldata = data.stack().reset_index().rename(columns={0:'value'})
# 
# print(ldata[:10],'\n')
# 
# pivoted = ldata.pivot('date','item','value')
# print(pivoted.head(),'\n')
# 
# ldata['value2'] = np.random.randn(len(ldata))
# print(ldata.head(),'\n')
# 
# pivoted2 = ldata.pivot('date','item')
# print(pivoted2.head(),'\n')
# 
# unstacked = ldata.set_index(['date','item']).unstack('item')
# print(unstacked.head())

df = pd.DataFrame({'key':['foo','bar','baz'],
                   'A':[1,2,3],
                   'B':[4,5,6],
                   'C':[7,8,9]})

print(df,'\n')

melted = pd.melt(df,['key'])
print(melted,'\n')

reshaped = melted.pivot(index='key', columns='variable', values='value').reset_index()
print(reshaped,'\n')

print(pd.melt(df, id_vars=['key'], value_vars=['A','B']))

print(pd.melt(df, value_vars=['A','B','C']))
print(pd.melt(df, value_vars=['key','A','B']))









































































