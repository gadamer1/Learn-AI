
# Simple linear regression

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1:].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression

linearRegression = LinearRegression()
linearRegression.fit(X_train,y_train)

pred = linearRegression.predict(X_test)

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,linearRegression.predict(X_train),color='blue')
plt.xlabel('annual experiences')
plt.ylabel('salary')
plt.show()