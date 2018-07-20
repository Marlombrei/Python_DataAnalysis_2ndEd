from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
from dask.array.chunk import arange

pd.options.display.width = 500
# # 
# df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
#                 'key2' : ['one', 'two', 'one', 'two', 'one'],
#                 'data1' : np.random.randn(5),
#                 'data2' : np.random.randn(5)})
#   
# print(df,'\n')
#  
# 'Slice'
# grouped = df['data1'].groupby(by=df['key1'], axis=0)
# print(grouped,'\n')
#  
# 'Apply'
# print(grouped.mean(),'\n')
#  
#  
# #means = df[['data1','data2']].groupby(by=[df['key1'], df['key2']], axis=0).mean()
# means = df[['data1']].groupby(by=[df['key1'], df['key2']], axis=0).mean()
# print(means,'\n')
# print(means.unstack(),'\n')
# 
# 
# states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
# years = np.array([2005, 2005, 2006, 2005, 2006])
# 
# groupedbyArrays = df['data1'].groupby(by=[states,years], axis=0).mean()
# print(groupedbyArrays,'\n')
# 
# 
# print(df.groupby(by='key1', axis=0).mean(),'\n')
# 
# print(df.groupby(['key1','key2']).sum(),'\n')
# 
# for name, group in df.groupby('key1'):
#     print(name,'\n')
#     print(group,'\n')
#     
#     
# for (k1,k2), group in df.groupby(['key1','key2']):
#     print((k1,k2),'\n')
#     print(group,'\n')
# 
# pieces = dict(list(df.groupby('key1')))
# print(pieces,'\n')


# print(df.dtypes,'\n')
# grouped = df.groupby(by=df.dtypes, axis=1)
# print(grouped.size())
# 
# for dtype, group in grouped:
#     print(dtype,'\n')
#     print(group,'\n')


# print(df.groupby(['key1','key2'])[['data2']].mean(),'\n')
# print(df.groupby(['key1','key2'])[['data1']].mean(),'\n')
# 
# s_grouped = df.groupby(['key1','key2'])['data2']
# print(s_grouped,'\n')
# print(s_grouped.mean(),'\n')
# 
# people = pd.DataFrame(np.random.randn(5,5),
#                       columns = list('abcde'),
#                       index = ['Joe','Steve','Wes','Jim','Travis'])
# people.iloc[2:3 , [1,2]] = np.nan
# print(people,'\n')
# 
# mapping = {'a': 'red', 'b': 'red', 'c': 'blue','d': 'blue', 'e': 'red', 'f' : 'orange'}
# 
# by_column = people.groupby(by=mapping, axis=1)
# print(by_column.sum(),'\n')
# 
# map_series = pd.Series(mapping)
# print(map_series,'\n')
# 
# print(people.groupby(map_series,axis=1).sum(),'\n')
# 
# print(people.groupby(len).sum(),'\n')
# 
# key_list = ['one','two','three','four','five']
# print(people.groupby([key_list]).min(),'\n')
# 
# print(people.groupby([len,key_list]).min(),'\n')

# columns = pd.MultiIndex.from_arrays(arrays=[['US','US','US','JP','JP'],
#                                             [1,3,5,1,3]],
#                                     names=['cty','tenor'])
# 
# hier_df = pd.DataFrame(np.random.randn(4,5), columns=columns)
# hier_df.index.name = 'Index'
# print(hier_df,'\n')
# 
# print(hier_df.groupby(axis=1, level='cty').count(),'\n')
# 
# 
# grouped = df.groupby(by='key1')
# print(grouped.quantile(0.9),'\n')
# 
# def peak_to_peak(arr):
#     return arr.max() - arr.min()
# 
# print(grouped.agg(peak_to_peak),'\n')
# print(grouped.aggregate(peak_to_peak),'\n')
# 
# print(grouped.describe(),'\n')

path = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\examples\tips.csv'
 
tips = pd.read_csv(path)
tips['tip_pct'] = tips['tip'] / tips['total_bill']
print(tips.head(10),'\n')
# 
# grouped = tips.groupby(['day','smoker'])
# print(grouped.mean(),'\n')
# grouped_pct = grouped[['tip_pct']]
# print(grouped_pct.mean(),'\n')
# 
# 
# print(grouped_pct.agg('mean'),'\n')
# 
# print(grouped_pct.agg(['mean','max','std', peak_to_peak]),'\n')
# 
# print(grouped_pct.agg([('foo','mean'),('bar', np.std)]),'\n')
# 
# functions = ['count','mean','max']
# 
# result = grouped['tip_pct','total_bill'].agg(functions)
# print(result,'\n')
# 
# print(result[['tip_pct']],'\n')
# 
# list_of_tuples = [('Marlom','mean'),('Silva',np.var)]
# print(grouped['tip_pct','total_bill'].agg(list_of_tuples),'\n')
# 
# func_dict = {'tip':[('marlom',np.max)], 'size':['sum']}
# print(grouped.agg(func_dict),'\n')
# 
# 
# func_dict2 = {'tip_pct':['min','max','mean','std'],
#               'size':'sum'}
# 
# print(grouped.agg(func_dict2),'\n')
# 
# print(tips.groupby(['day','smoker']).mean(),'\n')
# res = tips.groupby(['day','smoker'], as_index=False).mean()
# print(res,'\n')
# 
# def top(df, n=5, column='tip_pct'):
#     return df.sort_values(by=column)[-n:]
# 
# print(top(tips),'\n')
# 
# print(tips.groupby('smoker').apply(top),'\n')
# 
# print(tips.groupby(['smoker','day']).apply(top, n=1, column='total_bill'),'\n')
# 
# 
# result = tips.groupby('smoker')['tip_pct'].describe()
# print(result,'\n')
# print(result.unstack(),'\n')
# 
# print(tips.groupby('smoker').apply(top),'\n')
# print(tips.groupby('smoker', group_keys=False).apply(top),'\n')
# 
# 
# frame = pd.DataFrame({'data1':np.random.randn(1000),
#                       'data2':np.random.randn(1000)})
# 
# quartiles = pd.cut(frame.data1,4)
# print(quartiles[:10],'\n')
# 
# def get_stats(group):
#     return {'min':group.min(), 'max':group.max(), 'count':group.count(), 'mean':group.mean()}
# 
# grouped = frame.data2.groupby(quartiles)
# print(grouped.apply(get_stats).unstack(),'\n')

#Return Quantile number
# grouping = pd.qcut(frame.data1, 10, labels=False)
# print(grouping[:10],'\n')
# 
# grouped = frame.data1.groupby(grouping)
# print(grouped.apply(get_stats).unstack(),'\n')
# 
# s = pd.Series(np.random.randn(6))
# 
# s[::2] = np.nan
# print(s,'\n')
# 
# print(s.fillna(s.mean()),'\n')
# 
# states = ['Ohio', 'New York', 'Vermont', 'Florida', 'Oregon', 'Nevada', 'California', 'Idaho']
# group_key = ['East']*4 + ['West']*4
# print(group_key,'\n')
# data = pd.Series(np.random.randn(8), index=states)
# data[['Vermont','Nevada','Idaho']] = np.nan
# print(data,'\n')
# 
# print(data.groupby(group_key).mean(),'\n')
# 
# fill_mean = lambda g: g.fillna(g.mean())
# print(data.groupby(group_key).apply(fill_mean),'\n')
# 
# fill_values = {'East':0.5, 'West':-1}
# fill_func = lambda g: g.fillna(fill_values[g.name])
# print(data.groupby(group_key).apply(fill_func),'\n')


#Hearts, Spades, Clubs, Diamonds

# suits = list('HSCD')
# card_val = (list(range(1,11)) + [10]*3)*4
# base_names = ['A'] + list(range(2,11)) + ['J','K','Q']
# 
# cards = []
# 
# for suit in suits:
#     cards.extend(str(num)+suit for num in base_names)
# deck = pd.Series(card_val, index=cards)
# 
# 
# def draw(deck, n=5):
#     return deck.sample(n=n)
# 
# print(draw(deck))
# 
# 
# get_suit = lambda card: card[-1]
# print(deck.groupby(get_suit).apply(draw,n=2))
# 
# print(deck.groupby(get_suit, group_keys=False).apply(draw,n=2))


# df = pd.DataFrame({'category':list('aaaabbbb'),
#                    'data':np.random.randn(8),
#                    'weights':np.random.randn(8)})
# print(df,'\n')
# 
# grouped = df.groupby('category')
# print(grouped,'\n')
# 
# get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
# print(grouped.apply(get_wavg),'\n')


# path = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\examples\stock_px_2.csv'
# 
# close_px = pd.read_csv(path, parse_dates=True, index_col=0)
# print(close_px.tail(),'\n')
# 
# spx_corr = lambda x: x.corrwith(x['SPX']) #Pairwise correlation o each column with the SPX column
# rets = close_px.pct_change().dropna()     #Pct Change on close_px
# 
# get_year = lambda x: x.year               #function to extract the years Note: year is a datetime function
# 
# by_year = rets.groupby(get_year)          #group the pct changes by year
# print(by_year.apply(spx_corr),'\n')
# 
# print(by_year.apply(lambda g: g['AAPL'].corr(g['MSFT'])),'\n')
# 
# import statsmodels.api as sm
# 
# def regress(data,yvar,xvars):
#     Y = data[yvar]
#     X = data[xvars]
#     X['intercept'] = 1.
#     result = sm.OLS(Y,X).fit()
#     return result.params
# 
# print(by_year.apply(regress,'AAPL',['SPX']),'\n')


print(tips.pivot_table(['tip_pct'],index=['time','smoker'], columns=['day'],
                       aggfunc=len, fill_value=0.0, margins=True),'\n')

print(pd.crosstab(index=[tips.time, tips.day], columns=tips.smoker, margins=True))

























































































