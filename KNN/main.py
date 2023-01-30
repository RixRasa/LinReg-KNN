import np as np
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


#1. READING DATA AND SHOWING DATA
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', None)
dataF = pd.read_csv('cakes.csv')
dataF.insert(loc = 0, column='Id', value = np.arange(0,len(dataF),1))


#2. DATA ANALYSIS
print(dataF.head())
'''Data profiling'''
print(dataF.info());          print(); print()
'''Feature statistic'''
print(dataF.describe())
print(dataF.describe(include = [object]));          print(); print()

#3. DATA CLEASNSING(in this section we wont do anything because not data is missing/corruped that we need to replace or remove
'''instead of data cleasnsing we will use this section to visualise parameters'''
plt.scatter(dataF.flour, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('flour');plt.show()
plt.scatter(dataF.eggs, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('eggs');plt.show()
plt.scatter(dataF.sugar, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('sugar');plt.show()
plt.scatter(dataF.milk, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('milk');plt.show()
plt.scatter(dataF.butter, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('butter');plt.show()
plt.scatter(dataF.baking_powder, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('baking_powder');plt.show()

#4. FEATURE ENGINEERING
"""NORMALIZATION"""
for column in dataF.keys().values:
    if(isinstance(dataF.at[0,column],str)):
        continue
    if(column == "type" or column == "Id"):
        continue
    max = dataF[column].max(); min = dataF[column].min()
    for row in range(0, dataF.shape[0]):
        dataF.at[row,column] = (dataF.at[row,column] - min) / (max - min)
print(dataF.describe());            print();print()
"""ENCODING DATA, SELECTING COLUMNS FOR TRAINING AND PREDICTION"""
data_train = dataF.loc[:,['flour','eggs','sugar','milk','butter','baking_powder']]
y = dataF.loc[:,'type']

