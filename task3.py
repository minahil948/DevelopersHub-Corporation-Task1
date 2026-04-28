import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

df = pd.read_csv("heart_disease_uci.csv")

print(df.head())
print(df.info())

df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include=['object', 'str']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop(['id', 'dataset', 'num'], axis=1)

df = pd.get_dummies(df, drop_first=True)

print(df.describe())

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show()

sns.countplot(x='target', data=df)
plt.show()

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

print("AUC:", roc_auc_score(y_test, y_prob))

importance = model.coef_[0]
features = X.columns

for i in range(len(features)):
    print(features[i], ":", importance[i])