from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import json


# lst = list('marlombrei')
# print (lst, '\n')
# # 
# # for i in enumerate(lst):
# #     print ('%d : %s'%(i[0], i[1]))
# 
# map = {k:v for k,v in enumerate(lst)}
# print (map)
# 
# map2 = {}
# for k,v in enumerate(lst):
#     map2[k] = v
# print (map2)
# 
#     
# print (sorted('ferreira'))


lst1 = ['mar','lom','brei','marlom','marlombrei']
lst2 = ['alves','ferreira','silva']

# nomes = list(zip(lst1,lst2))
# print (nomes,'\n')
# 
# firstno, secondno = zip(*nomes)
# print (firstno,'\n')
# print (secondno,'\n')
# 
# lst3 = list('marlombrei')
# mapped = dict(enumerate(lst3))
# print (mapped)
# 
# keys = 10
# value = mapped.get(keys, 'Not there')
# 
# print (value)

lst1.extend(lst2)
#print (lst1)

by_letter = {}

# for word in lst1:
#     letter = word[0]
#     if letter not in by_letter:
#         by_letter[letter] = [word]
#     else:
#         by_letter[letter].append(word)
# print (by_letter)

for word in lst1:
    by_letter.setdefault(word[0],[]).append(word)
print (by_letter['m'],'\n')
for k in by_letter:
    print (k, ':', by_letter[k])


df = pd.DataFrame(by_letter['m'])
print (df,'\n')














































































































































