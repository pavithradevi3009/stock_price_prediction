import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Step 1: Fetch historical data
stock_symbol = 'AAPL'  # Change this to any stock symbol, e.g., 'MSFT'
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years of data

data = yf.download(stock_symbol, start=start_date, end=end_date)
data = data[['Close']].dropna()  # Use closing prices

# Prepare features: Day index as X (simple time-based feature)
data['Day'] = np.arange(len(data))

# Step 2: Train-test split (80% train, 20% test)
X = data[['Day']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 3: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on test set
test_predictions = model.predict(X_test)
mse = np.mean((y_test - test_predictions) ** 2)
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# Step 4: Predict next 30 days
future_days = 30
last_day = data['Day'].max()
future_X = np.array([[last_day + i] for i in range(1, future_days + 1)])
future_predictions = model.predict(future_X)

# Print predictions
print("\nPredicted Closing Prices for Next 30 Days:")
for i, pred in enumerate(future_predictions, 1):
    print(f"Day {i}: ${pred:.2f}")

# Step 5: Visualize
plt.figure(figsize=(12, 6))
plt.plot(data['Day'], data['Close'], label='Historical Close', color='blue')
plt.plot(X_test['Day'], test_predictions, label='Test Predictions', color='green', linestyle='--')
future_days_range = np.arange(last_day + 1, last_day + future_days + 1)
plt.plot(future_days_range, future_predictions, label='Future Predictions', color='red', linestyle='--')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Closing Price ($)')
plt.legend()
plt.grid(True)
plt.show()
