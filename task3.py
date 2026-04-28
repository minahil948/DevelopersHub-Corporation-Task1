import pandas as pd

data=pd.read_csv("heart_disease_uci.csv")

print(data.head(10))
print(data.columns)

categorical_cols = ['cp', 'thal', 'slope', 'ca']
df = pd.get_dummies(data, columns=categorical_cols, drop_first=True)