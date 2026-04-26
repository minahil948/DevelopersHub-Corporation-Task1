import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data=yf.download("AAPL", start="2015-01-01", end="2024-01-01",
                 auto_adjust=True,interval="1d")
print(data.head(10))
