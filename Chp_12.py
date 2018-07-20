from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
import pytz
from scipy.stats import percentileofscore
from test.test_quopri import QuopriTestCase
pd.options.display.width = 500

path = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\examples\stock_px_2.csv'

# values = pd.Series(['apple','orange','apple','apple']*2)
# print(values,'\n')
# print(values.count(),'\n')
# print(values.value_counts(),'\n')
# print(values.unique(),'\n')
  
# values = pd.Series([1,0,1,2]*2)
# values2 = pd.Series([0,1,0,2]*2)
# dim = pd.Series(['apple','orange','straw'], index=['A','B','C'])
# 
# print(dim.take(indices=values),'\n')
# print(dim.take(indices=values2),'\n')
  
# fruits = ['apple','orange','apple','apple']*2
# N = len(fruits)
# df = pd.DataFrame({'fruit':fruits,
#                    'basket_id':np.arange(N),
#                    'count':np.random.randint(3,15,size=N),
#                    'weight':np.random.uniform(0,4,size=N)},
#                    columns=['basket_id','fruit','count','weight'])
# print(df['fruit'],'\n')
#  
# fruit_cat = df['fruit'].astype(dtype='category')
# print(fruit_cat,'\n')
#  
# c_values = fruit_cat.values
# print(c_values,'\n')
# print(fruit_cat.cat.categories,'\n')
# print(fruit_cat.cat.codes,'\n')
# # 
# df['fruit'] = df['fruit'].astype(dtype='category')
# print(df.fruit,'\n')
# 
# my_cat = pd.Categorical(['foo','bar','baz','foo','bar'])
# print(my_cat,'\n')
# #my_cat_values = my_cat.values
# print(my_cat.categories,'\n')
# print(my_cat.codes,'\n')
# 
# categories = ['foo','bar','baz']
# codes=[0,1,2,0,0,1]
# my_cats = pd.Categorical.from_codes(codes=codes, categories=categories, ordered=False)
# print(my_cats,'\n')
# print(my_cats.as_ordered(),'\n')
# my_cats2 = pd.Categorical.from_codes(codes=codes, categories=categories, ordered=True)
# print(my_cats2,'\n')
# 
# 
# np.random.seed(12345)
# 
# draws = np.random.randn(1000)
# 
# print(np.round(a=draws[:5], decimals=4),'\n')
# 
# bins = pd.qcut(draws,4, labels=['Q1','Q2','Q3','Q4'])
# print(bins,'\n')
# print(bins.categories,'\n')
# print(bins.codes,'\n')
# 
# bins2 = pd.Series(bins, name='quartile')
# results = (pd.Series(draws).groupby(bins2).agg(['count','min','max']).reset_index())
# print(results,'\n')
# 
# print(results['quartile'],'\n')

# N = 10000000
# draws = pd.Series(np.random.randn(N))
# labels = pd.Series(['foo','bar','baz','qux']*(N//4))
# #===============================================================================
# # Converting labels to Categorical
# categories = labels.astype(dtype='category')
# #===============================================================================
# print(labels.memory_usage())
# print(categories.memory_usage())

# s = pd.Series(list('abcd')*2)
# print(s,'\n')
# cat_s = s.astype(dtype='category')
# print(cat_s,'\n')
# print(cat_s.cat.codes,'\n')
# 
# actual_catogories = list('abcde')
# cat_s2 = cat_s.cat.set_categories(actual_catogories)
# print(cat_s2,'\n')
# print(cat_s.value_counts(),'\n')
# print(cat_s2.value_counts(),'\n')
# 
# cat_s3 = cat_s[cat_s.isin(['a','b'])]
# print(cat_s3,'\n')
# print(cat_s3.cat.remove_unused_categories())

# cat_s = pd.Series(list('abcd')*2, dtype='category')
# print(cat_s,'\n')
# 
# print(pd.get_dummies(cat_s),'\n')
# 
# df = pd.DataFrame({'key':['a','b','c']*4,
#                    'value':np.arange(12.)})
# print(df,'\n')
# 
# g = df.groupby(by='key').value
# print(g.mean(),'\n')
# 
# print(g.transform(lambda x: x.mean()),'\n')
# print(g.transform('mean'),'\n')
# 
# print(g.transform(lambda x: x*2),'\n')
# print(g.transform(lambda x: x.rank(ascending=True)),'\n')
# 
# def normalize(x):
#     return (x - x.mean()) / x.std()
# 
# print(g.transform(normalize),'\n')
# print(g.apply(normalize),'\n')
# 
# print(g.transform('mean'),'\n')
# 
# normalized = (df['value'] - g.transform('mean')) / g.transform('std')
# print(normalized,'\n')


N = 15
times = pd.date_range('2017-05-20 00:00', periods=N, freq='1min')
df = pd.DataFrame({'time':times,
                   'value':np.arange(N)})

print(df,'\n')

print(df.set_index('time').resample('5min').count())

df2 = pd.DataFrame({'time':times.repeat(3),
                    'key':np.tile(['a','b','c'], reps=N),
                    'value':np.arange(N*3)})
print(df2,'\n')

time_key = pd.TimeGrouper('5min')
resampled = df2.set_index('time').groupby(['key',time_key]).sum()
print(resampled,'\n')

print(resampled.reset_index())

print(df,'\n')















































































































































































