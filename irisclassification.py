import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

df= pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\Data Analysis\Data cleaning\IRIS.csv")

df.shape
df.sample(5)
df.info()
df.isnull().sum()

df.columns.tolist()

df.corr()

df['species'].unique()
df['species'].replace('Iris-setosa',0, inplace= True)
df['species'].replace('Iris-versicolor',1, inplace= True)
df['species'].replace('Iris-virginica',2, inplace= True)

df['species']
df.head()
df.rename(columns= {'species':'Target'}, inplace = True)
x = df.iloc[:, 0:4]
y = df.iloc[:,-1]


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=5)
x_test.shape
x_train.shape

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
accuracy_score(y_test, y_pred)


#Hyperparameter Training GridsearchCV
#Number of trees in random forest
n_estimators = [20,60,100,120]

# Number of features to consider at every split
max_features = [0.2,0.6,1.0]

# Maximum number of levels in tree
max_depth = [2,8,None]

# Number of samples
max_samples = [0.5,0.75,1.0]

param_grid ={'n_estimators': n_estimators,'max_features':max_features,'max_depth':max_depth,'max_samples':max_samples}
print(param_grid)

rf = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

rf_grid = GridSearchCV(estimator= rf, param_grid= param_grid, cv =5, verbose= 2, n_jobs=-1)
rf_grid.fit(x_train,y_train)
rf_grid.best_params_
rf_grid.best_score_


















