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
dataF = pd.read_csv('fuel_consumption.csv')
dataF.insert(loc = 0, column='Id', value = np.arange(0,len(dataF),1))


#2. DATA ANALYSIS
print(dataF.head())
'''Data profiling'''
print(dataF.info());          print(); print()
'''Feature statistic'''
print(dataF.describe())
print(dataF.describe(include = [object]));          print(); print() #Standard deviation of 'Model Year' is 0.0 witch means every row for that atribute is same so we can ignore that column"""


#3. DATA CLEASNSING
'''Finding NaN values'''
print(dataF.loc[dataF.ENGINESIZE.isnull()].head())
print(dataF.loc[dataF.TRANSMISSION.isnull()].head())
print(dataF.loc[dataF.FUELTYPE.isnull()].head())
print(); print()
'''Filling NaN values with mean and most freq values or deleting row if there are low amount of them'''
dataF.ENGINESIZE = dataF.ENGINESIZE.fillna(dataF.ENGINESIZE.mean())
dataF.FUELTYPE = dataF.FUELTYPE.fillna(dataF.FUELTYPE.mode()[0])
dataF.TRANSMISSION = dataF.TRANSMISSION.fillna(dataF.TRANSMISSION.mode()[0])
print(dataF.info());          print(); print()
"""Correlation marix and all diagrams"""
'''plt.scatter(dataF.ENGINESIZE, dataF.CO2EMISSIONS,color = 'red',marker='o' ,edgecolors='black'); plt.title('ENGINESIZE');plt.show()
plt.scatter(dataF.CYLINDERS, dataF.CO2EMISSIONS,color = 'red',marker='o' , edgecolors='black'); plt.title('CYLINDERS'); plt.show()
plt.scatter(dataF.FUELCONSUMPTION_CITY , dataF.CO2EMISSIONS,color = 'red',marker='o' , edgecolors='black');  plt.title('FUELCONSUMPTION_CITY'); plt.show()
plt.scatter(dataF.FUELCONSUMPTION_HWY , dataF.CO2EMISSIONS, color = 'red',marker='o' ,edgecolors='black' );  plt.title('FUELCONSUMPTION_HWY'); plt.show()
plt.scatter(dataF.FUELCONSUMPTION_COMB , dataF.CO2EMISSIONS, color = 'red',marker='o' ,edgecolors='black' );  plt.title('FUELCONSUMPTION_COMB'); plt.show()
plt.scatter(dataF.FUELCONSUMPTION_COMB_MPG , dataF.CO2EMISSIONS,color = 'red',marker='o' ,  edgecolors='black');  plt.title('FUELCONSUMPTION_COMB_MPG');plt.show()
plt.scatter(dataF.MODELYEAR, dataF.CO2EMISSIONS,color = 'red',marker='o' ,edgecolors='black');  plt.title('MODELYEAR');plt.show()
plt.scatter(dataF.MAKE, dataF.CO2EMISSIONS,color = 'red',marker='o' ,edgecolors='black');  plt.title('MAKE');plt.show()
plt.scatter(dataF.MODEL, dataF.CO2EMISSIONS,color = 'red',marker='o' ,edgecolors='black');  plt.title('MODEL');plt.show()
plt.scatter(dataF.VEHICLECLASS, dataF.CO2EMISSIONS,color = 'red',marker='o' ,edgecolors='black');  plt.title('VEHICLECLASS');plt.show()
plt.scatter(dataF.TRANSMISSION, dataF.CO2EMISSIONS,color = 'red',marker='o' ,edgecolors='black');  plt.title('TRANSMISSION');plt.show()
plt.scatter(dataF.FUELTYPE, dataF.CO2EMISSIONS,color = 'red',marker='o' ,edgecolors='black');  plt.title('FUELTYPE');plt.show()
sb.heatmap(dataF.corr(), annot = True,square=True, fmt='.2f')
plt.show()'''


#4. FEATURE ENGINEERING
"""NORMALIZATION"""
for column in dataF.keys().values:
    if(isinstance(dataF.at[0,column],str)):
        continue
    if(column == "MODELYEAR" or column == "Id"):
        continue
    max = dataF[column].max(); min = dataF[column].min()
    for row in range(0, dataF.shape[0]):
        dataF.at[row,column] = (dataF.at[row,column] - min) / (max - min)
print(dataF.describe());            print();print()
"""ENCODING DATA, SELECTING COLUMNS FOR TRAINING AND PREDICTION"""
data_train = dataF.loc[:,['ENGINESIZE','CYLINDERS','FUELTYPE','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']]
y = dataF.loc[:,'CO2EMISSIONS']

ohe = OneHotEncoder(dtype = int, sparse = False)
fuel = ohe.fit_transform(data_train.FUELTYPE.to_numpy().reshape(-1,1))

data_train.drop(columns =['FUELTYPE'], inplace=True)
data_train = data_train.join(pd.DataFrame(data = fuel, columns = ohe.get_feature_names(['FUELTYPE'])))
print(data_train.head())


#5. MODEL TRAINING
'''Training , separating parameters, fitting, predicting'''
lr = LinearRegression()#built in model
X_train,X_test,y_train,y_test = train_test_split(data_train, y, train_size=0.7, random_state=123,shuffle=True) #separating
lr.fit(X_train,y_train) #training

labels_predicted = lr.predict(X_test) #predictions
ser_predicted = pd.Series(data = labels_predicted, name = 'Predicted', index = X_test.index)
res_df = pd.concat([X_test,y_test,ser_predicted], axis = 1)
print(res_df.head())

'''Mse error'''
mse = mean_squared_error(res_df['CO2EMISSIONS'], res_df['Predicted'])
print(mse)


#6. MAKING LINEARREGRESSION MODEL FROM SCRATCH
'''Creating model'''
class LinearRegressionGradientDescent:
    #M IS NUMBER OF INSTACES AND N IS NUMBER OF PARAMETERS
    def __init__(self):
        self.coeff = None
        self.features = None
        self.target = None
        self.mse_history = None

    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    def cost(self):
        predicted = self.features.dot(self.coeff)
        s = pow(predicted - self.target, 2).sum()
        return (0.5 / len(self.features)) * s

    def fit(self, features, target):
        self.features = features.copy(deep=True)
        self.features.insert(loc=0, column='c0', value=np.ones((len(features), 1)))
        self.coeff = np.zeros(shape= len(self.features.columns)).reshape(-1, 1) #COEF matrix dimensions (N + 1) x 1

        self.features = self.features.to_numpy() #FEATURES matrix dimensions M x (N + 1)
        self.target = target.to_numpy().reshape(-1, 1) #TARGET matrix dimensions M x 1

    def perform_gradient_descent(self, learning_rate, num_iterations=200):
        # We will remember history of MSE - costs
        self.mse_history = []
        for i in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
            self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    def gradient_descent_step(self, learning_rate):
        predicted = self.features.dot(self.coeff) #Dimension: M x 1
        #Features dimensions: M x (N + 1),
        #Features.T dimensions: (N + 1) x M,
        #(predicted - target) dimensions: M x 1,
        #Features.T [(N + 1) x M] x (predicted - target)[M x 1] = s [(N + 1) x 1]
        s = self.features.T.dot(predicted - self.target)
        gradient = (1. / len(self.features)) * s
        self.coeff = self.coeff - learning_rate * gradient
        return self.coeff, self.cost()

'''Using model training it and predicting'''
lrgd = LinearRegressionGradientDescent()
lrgd.fit(X_train, y_train)

learning_rates = np.array([[0.17], [0.005], [0.005], [0.005], [0.005], [0.005], [0.005], [0.005], [0.005], [0.005]])
coefs, mseCosts = lrgd.perform_gradient_descent(learning_rates, 300000)

y_predicted = lrgd.predict(X_test)
serY_predicted = pd.Series(data = y_predicted, name = 'Predicted', index = X_test.index)
res_dfF = pd.concat([X_test,y_test,serY_predicted], axis = 1)
print(res_dfF.head())

'''MSE ERROR, R2-ERROR and Coefficients'''
mse = mean_squared_error(res_dfF['CO2EMISSIONS'], res_dfF['Predicted'])
print("Mean square error: " , mse)
def r2score(y_pred, y):
    rss = np.sum((y_pred - y) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2
r2 = r2score(y_predicted, y_test)
print("R2 error: ", r2)
print("Coefs:" , coefs)