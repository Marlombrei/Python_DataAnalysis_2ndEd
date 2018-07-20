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
pd.options.display.max_rows = 10


class Employee:
    
    num_of_emps = 0
    raise_amount = 1.04 #this is the class variable
    
    def __init__(self, first, last, pay):#you must add the arguments you want to accept
        self.first = first #these are the Instance Variables
        self.last = last
        self.pay = pay
        self.email = first+'.'+last+'@company.com'
        Employee.num_of_emps += 1
        
    def fullName(self):#this is a regular method. Any method that takes the self as first argument is a regular method
        return '{} {}'.format(self.first,self.last)
    
    def apply_raise(self):#this is a regular method
        self.pay = int(self.pay * Employee.raise_amount)#looks like using self.raise_amount allows other sub-classes to change this value
    
    @classmethod
    def set_raise_amount(cls, amount):#this means that now we are working with the class and not the instance
        cls.raise_amount = amount
        
    @classmethod
    def from_string(cls, emp_str):#creating alternative constructors for special situations
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)

class Developer(Employee):
    
    def __init__(self,first, last, pay, prog_lang):
        super().__init__(first, last, pay) #this syntas can only be used for Python 3 and above
        self.prog_lang = prog_lang
        
class Manager(Employee):
    raise_amount = 1.10
    
    def __init__(self,first, last, pay, employees=None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees
            
    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)
    
    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)
    
    def print_emps(self):
        for emp in self.employees:
            print('-->',emp.fullName())         
    

 
emp1 = Developer('marlombrei','silva',50000, 'Python')
emp2 = Developer('sharon','ferguson',650000,'Java')

mgr1 = Manager('Naiane','Silva',1000000,[emp1])

print(emp1.email,'\n',emp1.prog_lang,'\n')
print(mgr1.email,'\n')
mgr1.add_emp(emp2)
mgr1.print_emps()








# print(Employee.num_of_emps,'\n')
# print(emp1.first,'\n')
# print(emp1.last,'\n')
# print(emp1.pay,'\n')
# print(emp1.email,'\n')
# print(emp1.fullName(),'\n')
# print(Employee.fullName(emp1),'\n')#only methods can be printed this way
# 
# 
# print(emp1.pay,'\n')
# emp1.apply_raise()
# #print(Employee.__dict__,'\n')
# Employee.set_raise_amount(1.15)
# emp1.raise_amount = 1.06
# print(Employee.raise_amount,'\n')
# print(emp1.raise_amount,'\n')
# print(emp2.raise_amount,'\n')
# 
# emp_str_3 = 'naiane-silva-1000000'
# emp3 = Employee.from_string(emp_str_3)
# print(emp3.fullName())
# print(emp3.email)











































































































































































