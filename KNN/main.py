import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder



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
"""plt.scatter(dataF.flour, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('flour');plt.show()
plt.scatter(dataF.eggs, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('eggs');plt.show()
plt.scatter(dataF.sugar, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('sugar');plt.show()
plt.scatter(dataF.milk, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('milk');plt.show()
plt.scatter(dataF.butter, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('butter');plt.show()
plt.scatter(dataF.baking_powder, dataF.type,color = 'red',marker='o' ,edgecolors='black'); plt.title('baking_powder');plt.show()
sb.heatmap(dataF.corr(), annot = True,square=True, fmt='.2f')"""

#4. FEATURE ENGINEERING
"""NORMALIZATION & ENCODING DATA, SELECTING COLUMNS FOR TRAINING AND PREDICTION"""
le = LabelEncoder()
dataF.type = le.fit_transform(dataF.type)
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


#5. MODEL TRAINING
'''Training , separating parameters, fitting, predicting'''
knn = KNeighborsClassifier() #built in model
X_train,X_test,y_train,y_test = train_test_split(data_train, y, train_size=0.8, random_state=133,shuffle=True) #separating
knn.fit(X_train, y_train) #training

labels_predicted = knn.predict(X_test) #predictions
ser_predicted = pd.Series(data = labels_predicted, name = 'Predicted', index = X_test.index)
res_df = pd.concat([X_test,y_test,ser_predicted], axis = 1)
print(res_df.head())

'''Mse error & accuracy'''
mse = mean_squared_error(res_df['type'], res_df['Predicted'])
print("Mean square error: " , mse)
acc = accuracy_score(y_test, labels_predicted)
print("Accuracy: ", acc)
