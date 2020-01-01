#importing neccsary modeule

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading data from dataseet

dataset = pd.read_csv("data/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,1].values

#spliting dataset into test and train set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

# now fit the modle using LR

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

# predicting via test data

y_pred = regression.predict(X_test)


# visualize the data via test data

plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regression.predict(X_train), color = 'green')
plt.title("YearsExperience vs Salary")
plt.xlabel("YearsExperience")
plt.ylabel("salary")
plt.legend()
plt.show()
