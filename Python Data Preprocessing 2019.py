# IEE 520: Fall 2019
# Python: data preprocessing
# Klim Drobnyh (klim.drobnyh@asu.edu)

# For compatibility with Python 2
from __future__ import print_function

import numpy as np
import pandas as pd

# to import different encoders
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

# To support plots
import matplotlib.pyplot as plt


print('1. Creating a new dataset')
data = pd.DataFrame({
            'x1': [1, 1, 6, 4],
            'x2': ['active', 'sedentary', 'sedentary', 'moderately'],
            'x3': ['high', 'normal', 'normal', 'low'],
        })
print('The dataframe:')
print(data)
print('The same dataframe, but as a numpy array:')
print(data.values)

print('2. Dealing with categorical data')
# There are several common ways to treat categorical data:
#   1. Label encoding
#   2. One hot encoding

print('2.1. Label encoding')
# If we have a feature that measures some quantity, 
# but expressed in non-numbers (e.g., x3), 
# we want to convert it in ordered way:
#   low -> 0;
#   medium -> 1;
#   high -> 2.

# We should specify order here:
encoder = OrdinalEncoder(categories=[['low', 'normal', 'high']])
encoder.fit(data['x3'].values.reshape(-1, 1))

print('Before transformation:')
print(data['x3'])

data['x3'] = encoder.transform(data['x3'].values.reshape(-1, 1))
print('After transformation:')
print(data['x3'])

print('2.2. One hot encoding')
# If we have a feature with different categories, 
# usually the ones that cannot be compared easily (e.g., x2), 
# we can use one hot encoding. 
# In that case, three different binary variables will be added:
#   x2_active;
#   x2_sedentary;
#   x2_moderately.

print('Before transformation:')
print(data)

data = pd.get_dummies(data, columns=['x2'])

print('After transformation:')
print(data)

print('3. Writing to and reading from .csv files')

# Writing to .csv file:
data.to_csv('custom_data.csv')

# Reading from .csv file:
# Note: you should check separator in 
# the datafile and specify it: 
# sep=',' for ",", sep='\t' for tab.
data2 = pd.read_csv('custom_data.csv', index_col=0)

# Let's compare them:
print('Original:')
print(data)
print('Loaded:')
print(data2)

print('4. Writing to and reading from Excel files')

# Writing to Excel file:
data.to_excel("custom_data.xlsx")

# Reading from Excel file:
data2 = pd.read_excel('custom_data.xlsx', index_col=0)

# Let's compare them:
print('Original:')
print(data)
print('Loaded:')
print(data2)

print('5. Splitting dataset to features and target')
# Here we assume that our target variable is x1.
y = data['x1'].values
del data['x1']
X = data.loc[:, data.columns != 'x1'].values
print(X)
print(y)