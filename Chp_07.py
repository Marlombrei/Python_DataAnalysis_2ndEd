from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle



# string_data = pd.Series(['aarvark','artichoke',np.nan,'avocado'])
# print(string_data,'\n')
# 
# print(string_data.isnull(),'\n')
# 
# string_data[0] = None
# print(string_data.isnull(),'\n')
# 
# 
# data = pd.Series([1,np.nan,3.5,np.nan,7])
# print(data,'\n')
# print(data.dropna(),'\n')
# print(data[data.notnull()])

# data = pd.DataFrame([[1,6.5,3],[1,np.nan,np.nan],
#                      [np.nan,np.nan,np.nan],[np.nan,6.5,3]])
# print(data,'\n')
# 
# cleaned = data.dropna(axis=0, how='all')#axis, how, thresh, subset, inplace)
# print(cleaned,'\n')

# df = pd.DataFrame(np.random.randn(7,3))
# print(df,'\n')
# 
# df.iloc[:4,1] = np.nan
# df.iloc[:2,2] = np.nan
# print(df,'\n')
# 
# print(df.dropna(),'\n')
# print(df.dropna(thresh=2),'\n')
# 
# print(df.fillna(value=0),'\n')# method, axis, inplace, limit, downcast)
# 
# print(df.fillna({1:0.5, 2:0.0}),'\n')
# 
# print(df.mean())
# print (df.fillna(df.mean(), axis=0))
# 
# print(df,'\n')

# data = pd.DataFrame({'k1':['one','two']*3 + ['two'],
#                      'k2':[1,1,2,3,3,4,4,]})
# 
# print(data,'\n')
# print(data.duplicated(),'\n')
# print(data.drop_duplicates(),'\n')
# 
# data['v1'] = range(7)
# print(data,'\n')
# 
# print(data.drop_duplicates(subset=['k1']),'\n')
# print(data.drop_duplicates(subset=['k1'], keep='last'),'\n')

# data = pd.DataFrame({'food':['bacon','pulled pork','bacon','Pastrami','corned beef','Bacon','pastrami','honey ham','nova Lox'],
#                      'ounces':[4,3,12,6,7.5,8,3,5,6]})
# 
# print(data,'\n')
# 
# meat_to_animal = {'bacon':'pig',
#                   'pulled pork':'pig',
#                   'pastrami':'cow',
#                   'corned beef':'cow',
#                   'honey ham':'pig',
#                   'nova lox':'salmon'}
# 
# #the below codes have been changed from the book so I could try different ways to achieve the same result.
# # lowercased = data['food'].str.lower()
# # print(lowercased,'\n')
# 
# data['food'] = data['food'].str.lower()
# data['animal'] = data['food'].map(meat_to_animal)
# print(data,'\n')
# 
# data['animal2'] = data.food.map(lambda x: meat_to_animal[x.lower()])
# print(data,'\n')
# 
# data['food'] = data.food.map(lambda x: x.capitalize())
# print(data,'\n')


data = pd.Series([1,-999,2,-999,-1000,3])
print(data,'\n')

print(data.replace(to_replace=-999, value=np.nan),'\n')

print(data)






















































































































