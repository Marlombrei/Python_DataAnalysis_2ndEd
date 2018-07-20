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
from sklearn.linear_model import LogisticRegression
#import statsmodels.api as sm
#import statsmodels.formula.api as smf

pd.options.display.width = 500

path = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\datasets\titanic\train.csv'
path2 = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\datasets\titanic\test.csv'
# def dnorm(mean, variance, size=1):
#     if isinstance(size, int):
#         size=size,
#     return mean + np.sqrt(variance) * np.random.randn(*size)
# 
# np.random.seed(12345)
# 
# N = 100
# X = np.c_[dnorm(0,0.4,size=N),
#           dnorm(0,0.6,size=N),
#           dnorm(0,0.2,size=N)]
# 
# eps = dnorm(0,0.1,size=N)
# beta = [0.1,0.3,0.5]
# 
# y = np.dot(X, beta) + eps
# print('\n\n')
# print(X[:5],'\n')
# print(y[:5],'\n')


train = pd.read_csv(path)
test = pd.read_csv(path2)
print(train[:4],'\n')
impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)
print('***Training Dataset*** \n',train.isnull().sum(),'\n')
print('***Testing Dataset*** \n',test.isnull().sum(),'\n')

train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)

predictors = ['Pclass','IsFemale','Age']
X_train = train[predictors].values
X_test = test[predictors].values
y_train = train['Survived'].values

print(X_train[:5],'\n') 
print(y_train[:5],'\n') 

model = LogisticRegression()
print(model.fit(X_train, y_train),'\n')

y_predict = model.predict(X_test)
print(y_predict[:10])
























































































































































































