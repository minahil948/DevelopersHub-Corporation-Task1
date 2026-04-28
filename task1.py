import matplotlib.pyplot as plt
import pandas as pd
import seaborn as ssn

dataset=pd.read_csv("Iris.csv") 
dfcleaned = dataset.copy()
sh=dfcleaned.shape 
print("shape is :", sh)
print("column names :",dfcleaned.columns) 
print(dfcleaned.head(10))
dfcleaned.info()
print(dfcleaned.describe())

#scatter plot

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
petallength= dfcleaned['PetalLengthCm']
petalwidth=dfcleaned['PetalWidthCm']
ax[0].scatter(
    petallength,petalwidth, color='green'

)
ax[0].set_xlabel('petal lenth')
ax[0].set_ylabel('petal width')

ax[0].set_title('relationship btw petals')



sepallength= dfcleaned['SepalLengthCm']
sepallwidth=dfcleaned['SepalWidthCm']
ax[1].scatter(
    sepallength,sepallwidth, color='green'

)
ax[1].set_xlabel('sepal lenth')
ax[1].set_ylabel('sepal width')

ax[1].set_title('relationship btw sepals')

plt.tight_layout()
plt.show()

##histogram

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
labels=['PetalLengthCm','PetalWidthCm','SepalLengthCm','SepalWidthCm']
colors=['red','green','steelblue','orange']

for col, c , ax in zip(labels, colors, axes):
    ax.hist(dfcleaned[col],bins=20,color=c, density=True)
    ax.set_title(col)

plt.tight_layout()
plt.show()

#boxplot

plt.figure(figsize=(10, 6))

ssn.boxplot(data=dfcleaned[['SepalLengthCm', 'SepalWidthCm',
                            'PetalLengthCm', 'PetalWidthCm']])

plt.title("Box Plots for All Features")
plt.xticks(rotation=20)
plt.show()

#pairplot

ssn.pairplot(dfcleaned, hue="Species")
plt.show()

