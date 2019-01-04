import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values

print('origin\nX', X,'\n\nY\n',Y,'\n')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features = [3], dtype=int)
X = onehotencoder.fit_transform(X).toarray()
print('onehot\nX', X,'\n')
X = X[: , 1:]
print('onehot\nX', X,'\n')

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print('training sets and Test sets\nX_train=\n', X_train,"\nX_test\n",X_test,'\nY_train=\n', Y_train,"\nY_test\n",Y_test,'\n')

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
