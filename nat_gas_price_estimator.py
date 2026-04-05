import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Load the data ---
df = pd.read_csv("Nat_Gas.csv")
df.columns = ["Date", "Price"]
df["Date"] = pd.to_datetime(df["Date"], dayfirst=False)
df = df.sort_values("Date").reset_index(drop=True)

# Days since start (for trend) and month (for seasonality)
df["Days"] = (df["Date"] - df["Date"].min()).dt.days
df["Month"] = df["Date"].dt.month

# --- Build features: trend + Fourier seasonal terms ---
def make_features(days_array, months_array):
    sin_term = np.sin(2 * np.pi * months_array / 12)
    cos_term = np.cos(2 * np.pi * months_array / 12)
    return np.column_stack([days_array, sin_term, cos_term])

X = make_features(df["Days"].values, df["Month"].values)
y = df["Price"].values

# --- Fit linear regression ---
model = LinearRegression()
model.fit(X, y)

# --- Price estimator function ---
def estimate_price(date_str):
    """
    Takes a date string (e.g. '2025-06-30') and returns
    the estimated natural gas price for that date.
    Works for past dates and up to 1 year into the future.
    """
    date = pd.to_datetime(date_str)
    days = (date - df["Date"].min()).days
    month = date.month
    features = make_features(np.array([days]), np.array([month]))
    price = model.predict(features)[0]
    return round(price, 2)

# --- Test examples ---
print("Price on 2023-06-30:", estimate_price("2023-06-30"))
print("Price on 2024-09-30:", estimate_price("2024-09-30"))
print("Price on 2025-03-31 (extrapolated):", estimate_price("2025-03-31"))
print("Price on 2025-09-30 (extrapolated):", estimate_price("2025-09-30"))

# --- Visualization ---
future_dates = pd.date_range(df["Date"].min(), periods=len(df) + 12, freq="ME")
future_days = (future_dates - df["Date"].min()).days.values
future_months = future_dates.month.values
future_prices = model.predict(make_features(future_days, future_months))

plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Price"], "o", color="steelblue", label="Actual Monthly Prices")
plt.plot(future_dates, future_prices, "-", color="darkorange", label="Model Estimate + Extrapolation")
plt.axvline(x=df["Date"].max(), color="gray", linestyle="--", label="Extrapolation starts here")
plt.title("Natural Gas Price Estimate (Oct 2020 – Sep 2025)")
plt.xlabel("Date")
plt.ylabel("Price (USD/MMBtu)")
plt.legend()
plt.tight_layout()
plt.savefig("nat_gas_price_chart.png", dpi=150)
plt.show()
print("Chart saved.")
