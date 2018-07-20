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


pd.options.display.width = 500

path = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\datasets\titanic\train.csv'
path2 = r'C:\Users\Marlombrei\Documents\Python_For_DataAnalysis\pydata-book-2nd-edition\datasets\titanic\test.csv'

# ints = np.ones(shape=10, dtype=np.uint16)
# print([ints],'\n')
# 
# floats = np.ones(shape=10, dtype=np.float32)
# print([floats],'\n')
# 
# print(np.issubdtype(arg1=ints.dtype, arg2=np.integer),'\n')
# print(np.issubdtype(arg1=floats.dtype, arg2=np.floating),'\n')
# 
# print(np.float32.mro(),'\n')
# 
# print(np.issubdtype(ints.dtype, np.number),'\n')

# arr = np.arange(8)
# print(arr,'\n')
# 
# print(arr.reshape(4,2),'\n')
# print(arr.reshape((4,2), order='C'),'\n')
# print(arr.reshape((4,2), order='F'),'\n')
# 
# print(arr.reshape((4,2)).reshape((2,4)),'\n')



# arr = np.arange(15)
# print(arr,'\n')
# 
# print(arr.reshape((5,-1)),'\n')
# 
# other_arr = np.ones((3,5))
# print(other_arr,'\n')
# print(other_arr.shape,'\n')
# print(arr.reshape(other_arr.shape),'\n')
# 
# arr = np.arange(12).reshape((3,4))
# print(arr.ravel(),'\n')
# print(arr.ravel('F'),'\n')



arr1 = np.array([[1,2,3],[4,5,6]])
print(arr1,'\n')
arr2 = np.array([[7,8,9],[10,11,12]])
print(arr2,'\n')

print("""
#===============================================================================
# Concatenate
#===============================================================================
""")

print(np.concatenate([arr1,arr2], axis=0),'\n')
print(np.concatenate([arr1,arr2], axis=1),'\n')

print("""
#===============================================================================
# Vstack & Hstack
#===============================================================================
""")

print(np.vstack((arr1,arr2)),'\n')
print(np.hstack((arr1,arr2)),'\n')

print("""
#===============================================================================
# Split
#===============================================================================
""")

arr = np.random.randn(5,2)
print(arr,'\n')

first, second, third = np.split(arr, [1,3])
print(first,'\n')
print(second,'\n')
print(third,'\n')

print("""
#===============================================================================
# Stacking helpers: r_ & c_
#===============================================================================
""")
arr = np.arange(6)
print(arr,'\n')
arr1 = arr.reshape((3,2))
print(arr1,'\n')
arr2 = np.random.randn(3,2)
print(arr2,'\n')

print(np.r_[arr1,arr2],'\n')
print(np.c_[arr1,arr2],'\n')
print(np.c_[np.r_[arr1,arr2], arr],'\n')


print("""
#===============================================================================
# Repeat: replicates each element in an array some number of times
#===============================================================================
""")

arr = np.arange(3)
print(arr,'\n')

print(np.tile(A=arr, reps=3))
print(np.repeat(a=arr, repeats=3, axis=0))
print(np.repeat(a=arr, repeats=[4,6,4], axis=0))
print('\n\n')

arr = np.random.randn(2,2)
print(arr,'\n')
print(np.repeat(a=arr, repeats=2, axis=0),'\n')
print(np.repeat(a=arr, repeats=2, axis=1),'\n')
print(np.repeat(a=arr, repeats=[2,3], axis=0),'\n')

print("""
#===============================================================================
# Tile: is a shortcut for "stacking copies of array" along an axis
#===============================================================================
""")

print(arr,'\n')

print(np.tile(A=arr, reps=2),'\n')
print(np.tile(A=arr, reps=(2,1)),'\n')
print(np.tile(A=arr, reps=(3,2)),'\n')

print("""
#===============================================================================
# Take and Put
#===============================================================================
""")

arr = np.arange(10)*100
print(arr,'\n')

inds = [7,1,2,6]
print(arr[inds],'\n')
print(arr.take(inds),'\n')
arr.put(inds,42)
print(arr,'\n')

inds = [2,0,2,1]
arr = np.random.randn(3,4)
print(arr,'\n')
print(arr.take(inds,axis=1),'\n')
print(arr.take(inds,axis=0),'\n')

print("""
#===============================================================================
# Broadcasting: describes how arithmetic works between arrays of different shapes
#===============================================================================
""")

arr = np.arange(5)
print(arr,'\n')
print(arr*4,'\n')

arr = np.arange(12).reshape((4,3))
print(arr,'\n')
print(arr.mean(axis=0),'\n')
print(arr - arr.mean(axis=0),'\n')

arr = np.random.randn(4,3)
print(arr,'\n')

row_means = arr.mean(1)
print(row_means,'\n')
print(row_means.shape,'\n')
print(row_means.reshape((4,1)),'\n')
demeaned = arr - row_means.reshape((4,1))
print(demeaned,'\n')
print(demeaned.mean(1),'\n')

print("""
#===============================================================================
# Setting Array values by Broadcasting
#===============================================================================
""")

arr = np.zeros((4,3))
print(arr,'\n')

arr[:] = 5
print(arr,'\n')

col = np.array([1,2,3,4])
arr[:] = col[:,np.newaxis]
print(arr,'\n')

arr[:2] = [[-1.37],[0.509]]
print(arr,'\n')

print("""
#===============================================================================
# Advanced ufunc Usage
#===============================================================================
""")

arr = np.arange(10)
print(arr,'\n')

print(np.add.reduce(arr),'\n')
print(arr.sum(),'\n')

np.random.seed(123456)
arr = np.random.randn(5,5)
print(arr,'\n')

arr[::2].sort(1)
print(arr[:,:-1] < arr[:,1:])

print(np.logical_and.reduce(arr[:,:-1] < arr[:,1:], axis=1),'\n')

arr = np.arange(15).reshape((3,5))
print(arr,'\n')

print(np.add.accumulate(arr,axis=1),'\n')

arr = np.arange(3).repeat([1,2,2])
print(arr,'\n')

print(np.multiply.outer(arr,np.arange(5)),'\n')

arr = np.arange(10)
print(arr,'\n')
print(np.add.reduceat(arr,[0,5,8]),'\n')

print("""
#===============================================================================
# More about Sorting
#===============================================================================
""")

arr = np.random.randn(6)
print(arr,'\n')
arr.sort()
print(arr,'\n')

arr = np.random.randn(3,5)
print(arr,'\n')
arr[:,0].sort()
print(arr,'\n')



print(arr[:,::-1],'\n')

print("""
#===============================================================================
# Indirect Sorts: argsort & lexsort
#===============================================================================
""")
values = np.array([5,0,1,3,2])
print(values,'\n')

indexer = values.argsort()
print(indexer,'\n')
print(values[indexer],'\n')


print("""
#===============================================================================
# Alternative Sort Algorithms
#===============================================================================
""")

values = np.array(['2:first','2:second','1:first','1:second','1:third'])
print(values,'\n')
key = np.array([2,2,1,1,1])
print(key,'\n')
indexer = key.argsort(kind='mergersort')
print(indexer,'\n')
print(values.take(indexer),'\n')


print("""
#===============================================================================
# Partially Sorting Arrays
#===============================================================================
""")

np.random.seed(12345)
arr = np.random.randn(20)
print(arr,'\n')
print(np.partition(arr,4),'\n')#, axis, kind, order))
indices = np.argpartition(a=arr, kth=3)
print(indices,'\n')
print(arr.take(indices),'\n')

arr = np.array([0,1,7,12,15])
print(arr,'\n')
print(arr.searchsorted(9),'\n')
print(arr.searchsorted([0,8,11,16]),'\n')

print("""
#===============================================================================
# Bin Data with .searchsorted()
#===============================================================================
""")


data = np.floor(np.random.uniform(0,10000,size=50))
print(data,'\n')
bins = np.array([0,100,1000,5000,10000])
labels = bins.searchsorted(data)
print(labels,'\n')

print(pd.Series(data).groupby(by=labels).mean(),'\n')


















































































































