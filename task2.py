import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data=yf.download("AAPL", start="2015-01-01", end="2024-01-01",
                 auto_adjust=True,interval="1d")
print(data.head(10))


data["target"]=data["Close"].shift(-1)
data = data.dropna()

x=data[["High","Low","Open","Volume"]]
y=data["target"]
X_train, X_test, Y_train ,Y_test=train_test_split(x,y,test_size=0.2)

split=int(len(data)*0.8)

X_train=x[:split]
X_test=x[split:]

Y_train=y[:split]
Y_test=y[split:]


model=LinearRegression()
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)

MSE=mean_squared_error(Y_test,y_pred)
Rsq=r2_score(Y_test,y_pred)

print("mean square error is: ",MSE)
print("value of R square is: ",Rsq)

