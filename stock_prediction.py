import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor 
df = pd.read_csv("dataset/TCS1.csv")

print(df.head())
print(df.tail())
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df.shape)


# Convert Date into date format
df['Date'] = pd.to_datetime(df['Date'])

# Sort data by date
df = df.sort_values('Date')


# Create tomorrow's closing price
df['Tomorrow_Close'] = df['Close'].shift(-1)

print(df[['Date', 'Close', 'Tomorrow_Close']].head(10))


# Remove last row
df = df.dropna()

# Select input features
features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Input
X = df[features]

# Target
y = df['Tomorrow_Close']

# Split data into training and testing

split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

print("\nTraining rows:", len(train))
print("Testing rows:", len(test))

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# Training inputs and target
X_train = train[features]
y_train = train['Tomorrow_Close']

# Testing inputs and target
X_test = test[features]
y_test = test['Tomorrow_Close']

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

model = LinearRegression()
model.fit(X_train , y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", mae)


rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Root Mean Squared Error:", rmse)

r2 = r2_score(y_test, predictions)

print("R2 Score:", r2)

print("\nPredicted Prices:")
print(predictions[:10])

result = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': predictions
})

print("\nActual vs Predicted:")
print(result.head(10))

plt.figure(figsize=(12, 6))

plt.plot(y_test.values, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.xlabel("Test Data")
plt.ylabel("Stock Price")

plt.title("Actual vs Predicted TCS Stock Price")

plt.legend()

plt.tight_layout()

plt.show()

# Create Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Random Forest
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\n----- RANDOM FOREST PERFORMANCE -----")

print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)
print("\n===== MODEL COMPARISON =====")

print("\nLinear Regression:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("\nRandom Forest:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

comparison = pd.DataFrame({
    'Actual Price': y_test.values,
    'Linear Regression': predictions,
    'Random Forest': rf_predictions
})

comparison.to_csv("model_comparison.csv", index=False)

print("\nModel comparison saved successfully.")

# Get latest stock data
latest_data = df[features].iloc[-1:]

# Latest closing price
latest_close = df['Close'].iloc[-1]

# Predictions
linear_prediction = model.predict(latest_data)[0]
rf_prediction = rf_model.predict(latest_data)[0]

print("\n===================================")
print("       TCS STOCK PRICE PREDICTION")
print("===================================")

print(f"Latest Closing Price : ₹{latest_close:.2f}")

print(f"Linear Regression   : ₹{linear_prediction:.2f}")

print(f"Random Forest       : ₹{rf_prediction:.2f}")

if rf_prediction > latest_close:
    print("Predicted Direction : UP")
elif rf_prediction < latest_close:
    print("Predicted Direction : DOWN")
else:
    print("Predicted Direction : NO CHANGE")

print("===================================")