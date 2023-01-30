import pandas as pd
import numpy as np
import math as m
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

"""SELECTING COLUMNS FOR TRAINING AND PREDICTION"""
data_train = dataF.loc[:,['flour','eggs','sugar','milk','butter','baking_powder']]
y = dataF.loc[:,'type']


#5. MODEL TRAINING
'''Training , separating parameters, predicting'''
knn = KNeighborsClassifier() #built in model
X_train,X_test,y_train,y_test = train_test_split(data_train, y, train_size=0.8, random_state=133,shuffle=True) #separating
knn.fit(X_train, y_train) #training

labels_predicted = knn.predict(X_test) #predictions
ser_predicted = pd.Series(data = labels_predicted, name = 'Predicted', index = X_test.index)
res_df = pd.concat([X_test,y_test,ser_predicted], axis = 1)
print(res_df.head(20))

'''Mse error & accuracy'''
mse = mean_squared_error(res_df['type'], res_df['Predicted'])
print("Mean square error: " , mse)
acc = accuracy_score(y_test, labels_predicted)
print("Accuracy: ", acc)


#6 MAKING KNN MODEL FROM SCRATCH
def distance(ins1, ins2):
    sum = 0
    for i in range(0, ins1.size):
        sum += (ins1[i] - ins2[i])**2
    return m.sqrt(sum)
class KNNModel(object):
    def __init__(self):
        self.features = None
        self.target = None
        self.N = None
    def fit(self, X, y):
        self.features = X.to_numpy()
        self.target = y.to_numpy()
        self.N = int(m.sqrt(self.features.shape[0]))
        self.N = 6
        #print(self.N)

    def predict(self, X_test):
        Xtest = X_test.to_numpy()
        ypredicted = []

        for i in range(0, Xtest.shape[0]):
            instance = Xtest[i]
            distances = []
            for j in range(0, self.features.shape[0]):
                instanceFeature = self.features[j]
                distances.append([distance(instance, instanceFeature),j])
            distances = sorted(distances , key = lambda l : l[0])
            cupcakeClosestNum = 0; muffinClosestNum = 0
            for k in range(0, self.N):
                indx = distances[k][1]
                if(0 == self.target[indx]):
                    cupcakeClosestNum += 1
                elif( 1 == self.target[indx]):
                    muffinClosestNum += 1
            if(cupcakeClosestNum > muffinClosestNum):
                ypredicted.append(0)
            else:
                ypredicted.append(1)
        return ypredicted

'''Training and predicting with our self made KNN model'''
knnModel = KNNModel()
knnModel.fit(X_train,y_train)

ypredicted = knnModel.predict(X_test)
ypredicted_ser = pd.Series(data = ypredicted, name = 'Predicted', index = X_test.index)
res_dfF = pd.concat([X_test,y_test,ypredicted_ser], axis = 1)
print(res_dfF.head(20))

'''Mse error & accuracy'''
mseM = mean_squared_error(res_dfF['type'], res_dfF['Predicted'])
print("Mean square error: " , mseM)
accM = accuracy_score(y_test, ypredicted)
print("Accuracy: ", accM)