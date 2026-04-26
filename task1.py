import pandas as pd

dataset=pd.read_csv("Iris.csv") 
dfcleaned=dataset.dropna() 
sh=dfcleaned.shape 
print("shape is :", sh)
print("column names :",dfcleaned.columns) 
print(dfcleaned.head(10))
print(dfcleaned.info())
print(dfcleaned.describe())