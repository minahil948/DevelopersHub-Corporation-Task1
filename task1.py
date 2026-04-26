import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Iris.csv") 
dfcleaned=dataset.dropna() 
sh=dfcleaned.shape 
print("shape is :", sh)
print("column names :",dfcleaned.columns) 
print(dfcleaned.head(10))
print(dfcleaned.info())
print(dfcleaned.describe())


fig ,ax =plt.subplots(1,2)
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