# GOAL: Identify a single team that has the greatest probability of winning in 2026

# String Values: Classifier (Team Name)
# Int Values: Year, Team_#, Seed, Elimination Round, TR_Rank, SOS_Rank, Luck_Rank
# Double Values: TR_Rating, SOS_Rating, Luck_Rating (CAN BE NEGATIVE)


# L1 Regularization will solve redundant features


import subprocess

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# URL
data = './combined_features.csv'


# Load Data
df = pd.read_csv(data)

# ROUND BASELINE
# Treat teams with ROUND > 0 as past tourneys (known outcomes)
# and ROUND == 0 (latest year) as the upcoming tourney (2026)
past_mask = df['ROUND'] > 0
future_mask = df['ROUND'] == 0

past = df[past_mask].copy()
future = df[future_mask].copy()

# Champ label : elimation ROUND == 1
past['CHAMPION'] = (past['ROUND'] == 1).astype(int)

feature_cols = [
    'YEAR',
    'TEAM_NO',
    'SEED',
    'ROUND',
    'TR_RANK',
    'SOS_RANK',
    'LUCK_RANK',
    'TR_RATING',
    'SOS_RATING',
    'LUCK_RATING'
]

x = past[feature_cols].values
y = past['CHAMPION'].values

#Split data 85-15
x_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression(
    #C=
    penalty='l1',
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced' # Accounting for class imbalance (few Champions)
)

# Train on training split
log_reg.fit(x_train_scaled, y_train)

# Evaluate on validation split
y_pred = log_reg.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Validation accuracy (85/15 split): {acc:.3f}")

# Re-train on all past data
x_all = x
y_all = y

x_all_scaled = scaler.fit_transform(x_all)
log_reg.fit(x_all_scaled, y_all)

# Identify the single team with highest prob
# of winning in 2026
x_future = future[feature_cols].values
x_future_scaled = scaler.transform(x_future)

# Predicted probaility of being Champ (class 1)
champ_probs = log_reg.predict_proba(x_future_scaled)[:, 1]
future['CHAMP_PROB'] = champ_probs

# Find the team with the maximum predicted probability
best_idx = np.argmax(champ_probs)
best_team = future.iloc[best_idx]['TEAM']
best_year = int(future.iloc[best_idx]['YEAR'])
best_prob = champ_probs[best_idx]

print("\nTeam with highest predicted probability of winning in 2026:")
print(f"Team: {best_team}")
print(f"Year (data row): {best_year}")
print(f"Predicted championship probability: {best_prob:.3f}")