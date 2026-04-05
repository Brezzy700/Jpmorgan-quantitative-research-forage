import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------
# STEP 1: Rebuild the price estimator from Task 1
# -------------------------------------------------------
df = pd.read_csv("Nat_Gas.csv")
df.columns = ["Date", "Price"]
df["Date"] = pd.to_datetime(df["Date"], dayfirst=False)
df = df.sort_values("Date").reset_index(drop=True)
df["Days"] = (df["Date"] - df["Date"].min()).dt.days
df["Month"] = df["Date"].dt.month

def make_features(days_array, months_array):
    sin_term = np.sin(2 * np.pi * months_array / 12)
    cos_term = np.cos(2 * np.pi * months_array / 12)
    return np.column_stack([days_array, sin_term, cos_term])

X = make_features(df["Days"].values, df["Month"].values)
y = df["Price"].values
model = LinearRegression()
model.fit(X, y)

def estimate_price(date_str):
    date = pd.to_datetime(date_str)
    days = (date - df["Date"].min()).days
    month = date.month
    features = make_features(np.array([days]), np.array([month]))
    return round(float(model.predict(features)[0]), 2)


# -------------------------------------------------------
# STEP 2: Contract Pricing Function
# -------------------------------------------------------
def price_storage_contract(
    injection_dates,       # list of date strings e.g. ["2024-01-31", "2024-02-29"]
    withdrawal_dates,      # list of date strings e.g. ["2024-06-30", "2024-07-31"]
    injection_rate,        # MMBtu per month you inject
    withdrawal_rate,       # MMBtu per month you withdraw
    max_storage_volume,    # max MMBtu that can be stored at any time
    storage_cost_per_month # cost per MMBtu per month to store
):
    """
    Prices a natural gas storage contract.

    Logic:
    - Buy (inject) gas on injection dates at estimated market price
    - Sell (withdraw) gas on withdrawal dates at estimated market price
    - Pay monthly storage costs for volume held in storage
    - Contract value = total revenue from sales - total purchase cost - total storage cost

    Parameters:
        injection_dates      : list of str, dates gas is purchased and stored
        withdrawal_dates     : list of str, dates gas is withdrawn and sold
        injection_rate       : float, MMBtu injected per date
        withdrawal_rate      : float, MMBtu withdrawn per date
        max_storage_volume   : float, max MMBtu storable at any time
        storage_cost_per_month: float, cost per MMBtu per month in storage

    Returns:
        dict with contract value and breakdown
    """

    injection_dates = sorted([pd.to_datetime(d) for d in injection_dates])
    withdrawal_dates = sorted([pd.to_datetime(d) for d in withdrawal_dates])

    # --- Validate volume constraints ---
    total_injected = injection_rate * len(injection_dates)
    total_withdrawn = withdrawal_rate * len(withdrawal_dates)

    if total_injected > max_storage_volume:
        print(f"WARNING: Total injection ({total_injected} MMBtu) exceeds max storage ({max_storage_volume} MMBtu).")
        print("Capping injections to max storage volume.")
        total_injected = max_storage_volume

    if total_withdrawn > total_injected:
        print(f"WARNING: Withdrawal ({total_withdrawn} MMBtu) exceeds injected volume ({total_injected} MMBtu).")
        print("Capping withdrawals to injected volume.")
        total_withdrawn = total_injected

    # --- Calculate purchase cost (injection) ---
    purchase_cost = 0
    for date in injection_dates:
        price = estimate_price(str(date.date()))
        volume = min(injection_rate, max_storage_volume - purchase_cost / price if purchase_cost > 0 else injection_rate)
        purchase_cost += price * injection_rate

    # --- Calculate revenue (withdrawal) ---
    revenue = 0
    for date in withdrawal_dates:
        price = estimate_price(str(date.date()))
        revenue += price * withdrawal_rate

    # --- Calculate storage cost ---
    # Storage cost = avg volume held * months held * cost per month
    # Approximate: volume stored for the period between first injection and last withdrawal
    first_injection = injection_dates[0]
    last_withdrawal = withdrawal_dates[-1]
    months_stored = (last_withdrawal.year - first_injection.year) * 12 + \
                    (last_withdrawal.month - first_injection.month)
    avg_volume = (total_injected + total_withdrawn) / 2  # average volume in storage
    total_storage_cost = avg_volume * months_stored * storage_cost_per_month

    # --- Contract Value ---
    contract_value = revenue - purchase_cost - total_storage_cost

    print("=" * 50)
    print("   NATURAL GAS STORAGE CONTRACT PRICING")
    print("=" * 50)
    print(f"  Injection dates     : {[str(d.date()) for d in injection_dates]}")
    print(f"  Withdrawal dates    : {[str(d.date()) for d in withdrawal_dates]}")
    print(f"  Injection rate      : {injection_rate} MMBtu/date")
    print(f"  Withdrawal rate     : {withdrawal_rate} MMBtu/date")
    print(f"  Max storage volume  : {max_storage_volume} MMBtu")
    print(f"  Storage cost        : ${storage_cost_per_month}/MMBtu/month")
    print("-" * 50)
    print(f"  Total Purchase Cost : ${purchase_cost:,.2f}")
    print(f"  Total Revenue       : ${revenue:,.2f}")
    print(f"  Total Storage Cost  : ${total_storage_cost:,.2f}")
    print(f"  Months in storage   : {months_stored}")
    print("-" * 50)
    print(f"  CONTRACT VALUE      : ${contract_value:,.2f}")
    print("=" * 50)

    return {
        "contract_value": round(contract_value, 2),
        "purchase_cost": round(purchase_cost, 2),
        "revenue": round(revenue, 2),
        "storage_cost": round(total_storage_cost, 2),
        "months_stored": months_stored
    }


# -------------------------------------------------------
# STEP 3: Test with sample inputs
# -------------------------------------------------------
if __name__ == "__main__":

    # Test 1: Buy in winter, sell in summer (classic seasonal play)
    print("\nTEST 1: Buy Jan-Feb, Sell Jun-Jul")
    result1 = price_storage_contract(
        injection_dates=["2024-01-31", "2024-02-29"],
        withdrawal_dates=["2024-06-30", "2024-07-31"],
        injection_rate=1000,
        withdrawal_rate=1000,
        max_storage_volume=5000,
        storage_cost_per_month=0.05
    )

    # Test 2: Buy in summer, sell in winter
    print("\nTEST 2: Buy Jun-Jul, Sell Nov-Dec")
    result2 = price_storage_contract(
        injection_dates=["2024-06-30", "2024-07-31"],
        withdrawal_dates=["2024-11-30", "2024-12-31"],
        injection_rate=1000,
        withdrawal_rate=1000,
        max_storage_volume=5000,
        storage_cost_per_month=0.05
    )

    # Test 3: Single injection, single withdrawal, future dates
    print("\nTEST 3: Single injection Mar 2025, withdrawal Sep 2025")
    result3 = price_storage_contract(
        injection_dates=["2025-03-31"],
        withdrawal_dates=["2025-09-30"],
        injection_rate=2000,
        withdrawal_rate=2000,
        max_storage_volume=5000,
        storage_cost_per_month=0.05
    )
