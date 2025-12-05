# GOAL: Identify a single team that has the greatest probability of winning in 2025
#       using past tournament data (ignoring 2025 outcomes in training).

# String Values: Classifier (Team Name)
# Int Values: Year, Team_#, Seed, Elimination Round, TR_Rank, SOS_Rank, Luck_Rank
# Double Values: TR_Rating, SOS_Rating, Luck_Rating (CAN BE NEGATIVE)

# L1 Regularization will solve redundant features


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# URL / path to data
data = './combined_features.csv'

# Load data
df = pd.read_csv(data)

# ---------------------------------------------------
# Split into (1) training years (< 2025) and (2) 2025
# ---------------------------------------------------
# Training data: all teams from years BEFORE 2025 with known outcomes
train_mask = (df['YEAR'] < 2025) & (df['ROUND'] > 0)
train_df = df[train_mask].copy()

# Future prediction target: teams in the 2025 tournament
future_mask = (df['YEAR'] == 2025)
future_df = df[future_mask].copy()

# ----------------------------------------------
# Build target label: CHAMPION (1) vs non-champ (0)
# ----------------------------------------------
# Champion = team eliminated in ROUND == 1 (the champion row in your encoding)
train_df['CHAMPION'] = (train_df['ROUND'] == 1).astype(int)

# ---------------------------------------------------
# Features: ints + doubles given in the specification
# ---------------------------------------------------
feature_cols = [
    'YEAR',       # int
    'TEAM_NO',    # int
    'SEED',       # int
    'ROUND',      # int (elimination round; known for past, 0 or unknown for 2025)
    'TR_RANK',    # int
    'SOS_RANK',   # int
    'LUCK_RANK',  # int
    'TR_RATING',  # double
    'SOS_RATING', # double
    'LUCK_RATING' # double
]

X = train_df[feature_cols].values
y = train_df['CHAMPION'].values

# -----------------
# Split data 85-15
# -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y  # preserve class balance
)

# -----------------
# Standardization
# -----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Logistic Regression Model
# -----------------------------
log_reg = LogisticRegression(
    penalty='l1',
    solver='liblinear',   # supports L1
    max_iter=1000,
    class_weight='balanced'  # handle class imbalance (few CHAMPION=1)
)

# Train on training split
log_reg.fit(X_train_scaled, y_train)

# Evaluate on validation split
y_pred = log_reg.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Validation accuracy (85/15 split, years < 2025): {acc:.4f}")

# -----------------------------
# Re-train on ALL training data
# -----------------------------
# Refit scaler on all training rows (years < 2025)
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X)
log_reg.fit(X_all_scaled, y)

# -----------------------------
# Predict the 2025 champion
# -----------------------------
if not future_df.empty:
    X_future = future_df[feature_cols].values
    X_future_scaled = scaler.transform(X_future)

    # Probability of being champion (class 1)
    champ_probs = log_reg.predict_proba(X_future_scaled)[:, 1]
    future_df['CHAMP_PROB'] = champ_probs

    # Find the team with the highest probability in 2025
    best_idx = np.argmax(champ_probs)
    best_team = future_df.iloc[best_idx]['TEAM']
    best_year = int(future_df.iloc[best_idx]['YEAR'])
    best_prob = champ_probs[best_idx]

    print("\nPredicted 2025 champion (using only data from years < 2025):")
    print(f"Team: {best_team}")
    print(f"Year: {best_year}")
    print(f"Predicted championship probability: {best_prob:.4f}")
else:
    print("\nNo rows found for YEAR == 2025 in the dataset.")
