from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import json
import requests


# path = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\examples\ex1.csv'
# 
# frame = pd.read_csv(path)
# print(frame)
# 
# frame.to_pickle('pickle_frame')
# frame_p = pd.read_pickle('pickle_frame')
# 
# print(frame_p)
# 
# frame2 = pd.DataFrame({'a':np.random.randn(100)})
# store = pd.HDFStore('mydata.h5')
# store['obj1'] = frame2
# store['obj1_col'] = frame2['a']
# print(store)
# 
# print(store['obj1'])
# 
# 
# path = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\examples\ex1.xlsx'
# 
# xlsx = pd.ExcelFile(path)
# 
# print(pd.read_excel(xlsx, 'Sheet1'))

# url = r'https://api.github.com/repos/pandas-dev/pandas/issues'
# resp = requests.get(url)
# print(resp)
# 
# data = resp.json()
# print(data[0]['title'])
# 
# issues = pd.DataFrame(data, columns=['number','title','labels','state'])
# print(issues)

import sqlite3

query = '''
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
 c REAL,        d INTEGER
);'''

con = sqlite3.connect('mydata.sqlite')
con.execute(query)
con.commit()






































































