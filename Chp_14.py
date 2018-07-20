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
import json
from collections import defaultdict, Counter

pd.options.display.width = 500
pd.options.display.max_rows = 10





#records = [json.loads(line) for line in open(path)]

# time_zones = [rec['tz'] for rec in records if 'tz' in rec]
# print(time_zones[:10],'\n')
# 
# def get_counts(sequence):
#     counts = {}
#     for x in sequence:
#         if x in counts:
#             counts[x] += 1
#         else:
#             counts[x] = 1
#     return counts
# 
# print(get_counts(time_zones),'\n')
# 
# def get_counts2(sequence):
#     counts = defaultdict(int) #values will initialize to 0
#     for x in sequence:
#         counts[x] +=1
#     return counts
# 
# print(get_counts2(time_zones),'\n')
# 
# def top_counts(count_dict, n=10):
#     value_key_pairs = [(count,tz) for tz, count in count_dict.items()]
#     value_key_pairs.sort()
#     return value_key_pairs[-n:]
# 
# counts = get_counts(time_zones)
# top_ten = top_counts(counts, n=10)
# print(top_ten)
# 
# test = [(k,v) for k,v in counts.items()]
# print(test)
# 
# counts2 = Counter(time_zones)
# print(counts2.most_common(10))


# frame = pd.DataFrame(records)
# print(frame.info(),'\n')
# 
# print(frame[:10])
# tz_counts = frame['tz'].value_counts()
# print(tz_counts[:10],'\n')
# 
# clean_tz = frame['tz'].fillna(value='Missing')
# clean_tz[clean_tz==''] = 'Unknown'
# tz_counts = clean_tz.value_counts()
# print(tz_counts[:10],'\n')
# 
# #print(frame['a'][51])
# 
# results = pd.Series([x.split()[0] for x in frame.a.dropna()])
# #print(results)
# 
# #print(results.value_counts())
# 
# cframe = frame[frame.a.notnull()]
# cframe['os'] = np.where(cframe['a'].str.contains('Windows'),'Windows','Non-Windows')
# #print(cframe['os'])
# 
# by_tz_os = cframe.groupby(['tz','os'])
# agg_counts = by_tz_os.size().unstack().fillna(0)
# print(agg_counts[:10],'\n')
# 
# indexer = agg_counts.sum(axis=1).argsort()
# print(indexer[-10:],'\n')
# 
# count_subset = agg_counts.take(indexer[-10:])
# print(count_subset.stack().reset_index())
# 
# #print(agg_counts.sum(axis=1).nlargest(10))


#===============================================================================
# MovieLens 1M Dataset
#===============================================================================
# movies_file = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\datasets\movielens\movies.dat'
# ratings_file = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\datasets\movielens\ratings.dat'
# users_file = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\datasets\movielens\users.dat'
# unames = ['user_id','gender','age','occupation','zip']
# users = pd.read_table(users_file, sep='::', header=None, names=unames, engine='python')
# #print(users,'\n')
# 
# rnames = ['user_id','movie_id','rating','timestamp']
# ratings = pd.read_table(ratings_file, sep='::', header=None, names=rnames, engine='python')
# #print(ratings,'\n')
# 
# mnames = ['movie_id','title','genres']
# movies = pd.read_table(movies_file, sep='::', header=None, names=mnames, engine='python')
# #print(movies,'\n')
# 
# #pd.merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator)
# data = pd.merge(pd.merge(ratings, users), movies)
# print(data,'\n')
# 
# mean_ratings = data.pivot_table('rating',index='title',columns='gender',aggfunc='mean')
# print(mean_ratings,'\n')
# 
# # merge(left, right, how='inner', on=None, left_on=None, right_on=None, 
# #     left_index=False, right_index=False, sort=False, 
# #     suffixes=('_x', '_y'), copy=True, indicator=False):
# 
# ratings_by_title = data.groupby('title').size()#how many times the title was shown on the data?
# print(ratings_by_title[:10],'\n')
# active_titles = ratings_by_title.index[ratings_by_title >= 250]
# print(active_titles[:10],'\n')
# 
# #Select rows on the index
# mean_ratings = mean_ratings.loc[active_titles]
# print(mean_ratings,'\n')
# 
# top_female_ratings = mean_ratings.sort_values(by='M', ascending=False)
# print(top_female_ratings,'\n')
# 
# mean_ratings['Diff'] = mean_ratings['F'] - mean_ratings['M']
# print(mean_ratings,'\n')
# top_diff = mean_ratings.sort_values(by='Diff', ascending=False)
# print(top_diff,'\n')
# 
# rating_std_by_title = data.groupby('title')['rating'].std()
# rating_std_by_title = rating_std_by_title.loc[active_titles]
# print(rating_std_by_title.sort_values(ascending=True),'\n')


#===============================================================================
# US Baby Names 1880-2010
#===============================================================================
# yob_1880 = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\datasets\babynames\yob1880.txt'
# 
# names1880 = pd.read_csv(yob_1880, names=['name','sex','births'])
# #print(names1880,'\n')
# 
# #print(names1880.groupby('sex')['births'].sum())
# 
# years = range(1880,2011)
# pieces = []
# columns = ['name','sex','births']
# 
# for year in years:
#     path = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\datasets\babynames\yob%d.txt'%year
#     frame = pd.read_csv(path, names=columns)
#     frame['year'] = year
#     pieces.append(frame)
#     
# 
# names = pd.concat(pieces, ignore_index=True)
# 
# total_births = names.pivot_table('births',index='year',columns='sex', aggfunc=sum)
#print(total_births,'\n')


# def add_prop(group):
#     group['prop'] = group.births / group.births.sum()
#     return group
# 
# names = names.groupby(by=['year','sex']).apply(add_prop)
# #print(names,'\n')
# # 
# names['marlom'] = names.births / names.births.sum()
# print(names)
#  
# print(names.groupby(['year','sex']).prop.sum())
#  
# def get_top1000(group):
#     return group.sort_values(by='births', ascending=False)[:1000]
# # 
# top_1000 = names.groupby(['year','sex']).apply(get_top1000)
# top_1000.reset_index(inplace=True, drop=True)
# print(top_1000[:10])
# 
# boys = top_1000[top_1000.sex == 'M']
# girls = top_1000[top_1000.sex == 'F']
# 
# 
# 
# total_births = top_1000.pivot_table('births', index='year',columns='name',aggfunc=sum)
# print(total_births.info())
# 
# subset = total_births[['John','Harry','Mary','Marilyn']]
# #subset.plot(subplots=True, grid=False, title='Number of birhs per year')
# 
# table = top_1000.pivot_table('prop', index='year', columns='sex',aggfunc=sum)
# #table.plot(yticks=np.linspace(0,1.2,13), xticks=range(1880,2020,10))
# #plt.show()
# 
# df = boys[boys.year == 2010]
# print(df,'\n')
# 
# prop_cumsum = df.sort_values(by='prop', ascending=False)['prop'].cumsum()
# print(prop_cumsum,'\n')
# 
# print(prop_cumsum.values.searchsorted(0.5) + 1)
# 
# b_1900 = boys[boys.year == 1900]
# p_csum_1900 = b_1900.sort_values(by='prop', ascending=False)['prop'].cumsum()
# print(p_csum_1900.values.searchsorted(0.5) + 1)
# 
# 
# def get_quantile_count(group, q=0.5):
#     group = group.sort_values(by='prop', ascending=False)
#     return group['prop'].cumsum().values.searchsorted(q) + 1
# 
# diversity = top_1000.groupby(['year','sex']).apply(get_quantile_count)
# print(diversity,'\n')
# print(diversity.unstack('sex'),'\n')


# get_last_letter = lambda x: x[-1]
# last_letters = names.name.map(get_last_letter)
# last_letters.name = 'last_letter'
# #print(last_letters,'\n')
# 
# table = names.pivot_table('births', index=last_letters, columns=['sex','year'], aggfunc=sum)
# print(table['F'][1910],'\n')
# 
# subtable = table.reindex(columns=[1910,1960,2010], level='year') #reindex on Columns?
# print(subtable,'\n')
# print(subtable.sum(),'\n')
# 
# letter_prop = subtable / subtable.sum()
# print(letter_prop,'\n')
# 
# dny_ts = letter_prop.loc[['d','n','y'], 'M'].T
# print (dny_ts.head())
# 
# 
# 
# all_names = pd.Series(top_1000.name.unique())
# lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
# print(lesley_like)
# 
# filtered = top_1000[top_1000.name.isin(lesley_like)]
# print(filtered)
# 
# print(filtered.groupby('name')['births'].sum())
# 
# table = filtered.pivot_table('births', index='year', columns='sex',aggfunc='sum')
# table = table.div(table.sum(1), axis=0)
# print(table.tail())

#===============================================================================
# USDA Food Database
#===============================================================================
path = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\datasets\usda_food\database.json'
db = json.load(open(path))

info_keys = ['description','group','id','manufacturer']
info = pd.DataFrame(db,columns=info_keys)
print(info)

df_lst = []
for i in range(len(db)):
    nut = pd.DataFrame(db[i]['nutrients'])
    nut['id'] = db[i]['id']
    df_lst.append(nut)
nutrients = pd.concat(df_lst)

#drop duplicates
#print(nutrients.duplicated().sum())
nutrients = nutrients.drop_duplicates()

col_mapping = {'description':'food',
                'group':'fgroup'}

info = info.rename(columns=col_mapping, copy=False)

col_map = {'description':'nutrient',
           'group':'nutgroup'}
nutrients = nutrients.rename(columns=col_map, copy=False)
print(nutrients,'\n')

ndata = pd.merge(nutrients,info, on='id', how='outer')
print(ndata.info(),'\n')
print(ndata,'\n')

result = ndata.groupby(['nutrient','fgroup'])['value'].quantile(0.5)
result['Zinc, Zn'].sort_values().plot(kind='barh')
plt.show()








































































































































