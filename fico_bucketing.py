import pandas as pd
import numpy as np

# -------------------------------------------------------
# STEP 1: Load data
# -------------------------------------------------------
df = pd.read_csv("Task_3_and_4_Loan_Data.csv")
df = df[["fico_score", "default"]].dropna().sort_values("fico_score").reset_index(drop=True)

print("FICO range: {} to {}".format(df["fico_score"].min(), df["fico_score"].max()))
print("Total records:", len(df))
print("Default rate: {:.1%}".format(df["default"].mean()))

# -------------------------------------------------------
# STEP 2: Compress to unique FICO score groups (much faster)
# -------------------------------------------------------
grouped = df.groupby("fico_score")["default"].agg(["sum", "count"]).reset_index()
grouped.columns = ["fico_score", "defaults", "total"]

scores   = grouped["fico_score"].values   # unique FICO scores
k_arr    = grouped["defaults"].values     # defaults per score
n_arr    = grouped["total"].values        # records per score
M        = len(scores)                    # number of unique score values

# Cumulative sums over unique score groups
cum_k = np.concatenate([[0], np.cumsum(k_arr)])
cum_n = np.concatenate([[0], np.cumsum(n_arr)])

def range_ll(i, j):
    """Log-likelihood for unique score groups i to j-1."""
    k = cum_k[j] - cum_k[i]
    n = cum_n[j] - cum_n[i]
    if n == 0 or k == 0 or k == n:
        return 0.0
    p = k / n
    return k * np.log(p) + (n - k) * np.log(1 - p)

# -------------------------------------------------------
# STEP 3: DP over unique score groups
# -------------------------------------------------------
def find_optimal_buckets(n_buckets):
    NEG_INF = -np.inf
    dp    = np.full((n_buckets + 1, M + 1), NEG_INF)
    split = np.zeros((n_buckets + 1, M + 1), dtype=int)
    dp[0][0] = 0.0

    for b in range(1, n_buckets + 1):
        for i in range(b, M + 1):
            best_val = NEG_INF
            best_j   = b - 1
            for j in range(b - 1, i):
                if dp[b-1][j] == NEG_INF:
                    continue
                val = dp[b-1][j] + range_ll(j, i)
                if val > best_val:
                    best_val = val
                    best_j   = j
            dp[b][i]    = best_val
            split[b][i] = best_j

    # Backtrack
    idx     = M
    indices = []
    for b in range(n_buckets, 0, -1):
        indices.append(split[b][idx])
        idx = split[b][idx]
    indices = sorted(indices)

    # FICO boundaries from indices
    boundaries = [scores[i] for i in indices[1:]]

    print("\n" + "="*60)
    print(f"  OPTIMAL FICO BUCKETS  (n_buckets={n_buckets})")
    print("="*60)
    print(f"  Bucket boundaries (FICO): {boundaries}")

    full_bounds = [scores[0]] + boundaries + [scores[-1] + 1]
    bucket_info = []

    print(f"\n  {'Rating':<8} {'FICO Range':<20} {'Records':<10} {'Defaults':<10} {'Default%'}")
    print(f"  {'-'*60}")

    for rating, (lo, hi) in enumerate(zip(full_bounds[:-1], full_bounds[1:]), start=1):
        mask  = (df["fico_score"] >= lo) & (df["fico_score"] < hi)
        total = mask.sum()
        defs  = df.loc[mask, "default"].sum()
        pct   = defs / total if total > 0 else 0
        print(f"  {rating:<8} {str(lo)+' – '+str(int(hi)-1):<20} {total:<10} {defs:<10} {pct:.1%}")
        bucket_info.append((rating, lo, hi))

    print("="*60)

    def rating_map(fico_score):
        for rating, lo, hi in bucket_info:
            if lo <= fico_score < hi:
                return rating
        return n_buckets

    return boundaries, rating_map

# -------------------------------------------------------
# STEP 4: Run bucketing
# -------------------------------------------------------
boundaries_5, rate_5 = find_optimal_buckets(n_buckets=5)
boundaries_3, rate_3 = find_optimal_buckets(n_buckets=3)

# -------------------------------------------------------
# STEP 5: Test the rating map
# -------------------------------------------------------
print("\nSAMPLE FICO → RATING (5 buckets, 1 = best credit):")
print(f"  {'FICO':<10} {'Rating'}")
print(f"  {'-'*20}")
for fico in [450, 530, 570, 610, 650, 690, 730, 780, 820]:
    print(f"  {fico:<10} {rate_5(fico)}")
