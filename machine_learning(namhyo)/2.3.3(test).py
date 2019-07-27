#%%
from IPython.display import display
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
#%matplotlib inline
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
X,y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train,y_test = train_test_split(X,y,random_state = 0)
lr = LinearRegression().fit(X_train,y_train)

ridge = Ridge(alpha=100000).fit(X_train,y_train)
print("훈련세트점수 : {:.2f}".format(ridge.score(X_train,y_train)))
print("테스트세트점수 : {:.2f}".format(ridge.score(X_test,y_test)))