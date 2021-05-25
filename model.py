# importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

# This is a small flask ml web application

#Loading the data
data = pd.read_csv('advertising.csv')

# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)
score=r2_score(y_test,y_pred)

# new_data=[[50,25,25]]
# print(regressor.predict(new_data))

f = open('sales-prediction.pkl', 'wb')
pickle.dump(regressor, f)
f.close()