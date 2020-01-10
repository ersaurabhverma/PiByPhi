Installing:
For Windows:
 	pip install PiByPhi
For Linux:
	pip3 install PiByPhi


import pandas as pd
from PiByPhi import PiByPhiClassifier as pbpc
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


data=pd.read_csv('iris.csv')
X_train,X_test,y_train,y_test=train_test_split(data[["sepal.length","sepal.width","petal.length","petal.width"]],data["variety"],test_size=0.5,random_state=42)
model=pbpc()
model.fit(X_train,y_train)
model.evaluation(X_test,y_test,all_models=True)
