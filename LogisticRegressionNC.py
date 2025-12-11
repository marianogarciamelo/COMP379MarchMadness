# GOAL: Identify a single team that has the greatest probability of winning in 2026

# String Values: Classifier (Team Name)
# Int Values: Seed, Elimination Round, TR_Rank, SOS_Rank, Luck_Rank
# Double Values: TR_Rating, SOS_Rating, Luck_Rating (CAN BE NEGATIVE)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# URL
data = './combined_features.csv'
# # Load Data
# df = pd.read_csv(data)

def create_features(df):
    # Create meaningful features while avoiding data leakage (ROUND-based)
    # Only use info that would be available BEFORE the tournament starts

    df = df.copy()

    # Feature Columns
    feature_cols = [
    'SEED', # Tourney Seed (1-16)
    'TR_RANK', # Team Rating Rank
    'SOS_RANK', # Strength Of Schedule Rank
    'LUCK_RANK', # Close Game Rank
    'TR_RATING', # Team Rating
    'SOS_RATING', # Strength Of Schedule Rating
    'LUCK_RATING' # Close Game Rating
    ]


    return df, feature_cols



def prepare_data(df, val_year=2025, exclude_recent=False):
    # Prepare data for modeling
    df, feature_cols = create_features(df)

    # Define target variable
    df['CHAMP'] = (df['ROUND'] == 1).astype(int)
    
    historical_df = df[df['ROUND'] > 0].copy()

    if val_year is None:
        val_year = historical_df['YEAR'].max()

    val_df = historical_df[historical_df['YEAR'] == val_year].copy()

    if exclude_recent:
        train_df = historical_df[historical_df['YEAR'] < val_year].copy()
    else:
        train_df = historical_df
    
    return train_df, val_df, feature_cols
        

def train(X_train, Y_train):
    # Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Logistic Regression with L1 Regularization
    log_reg = LogisticRegression(
        penalty='l1',
        solver='liblinear', # 'liblinear' supports L1 penalty
        max_iter=1000,
        class_weight='balanced' # Accounting for class imbalance (few Champions)
    )

    print("Training Logistic Regression Model...")
    log_reg.fit(X_train_scaled, Y_train)
    return log_reg, scaler


def evaluate(model, scaler, X_test, Y_test, team_names=None):
    # Evaluate Model with multiple metrics
    
    #  Standardize Validation Features
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    Y_pred = model.predict(X_test_scaled)
    Y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Validation Accuracy: {accuracy:.3f}")

    report = classification_report(Y_test, Y_pred, target_names=['Not Champion', 'Champion'])
    print("Classification Report:", report)
    if team_names is not None:
        top_k = min(10, len(Y_pred_prob))
        top_indices = np.argsort(Y_pred_prob)[-top_k:][::-1]

        print(f"Top {top_k} Teams by Predicted Championship Probability:")
        for idx in top_indices:
            prob = Y_pred_prob[idx]
            actual = "Champion" if Y_test.iloc[idx] == 1 else ""
            print(f"Team: {team_names.iloc[idx]:30s}, Predicted Probability: {prob:.3f} {actual}")
        

    return Y_pred, Y_pred_prob


if __name__ == "__main__":
    # Load Data
    df = pd.read_csv(data)

    # Train on years before 2025, Validate on 2025
    train_df, val_df, feature_cols = prepare_data(df, val_year=None, exclude_recent=True)

    # Train and Predict
    print(f"Training on years: {train_df['YEAR'].min()} - {train_df['YEAR'].max()}")
    print(f"Validating on year: {val_df['YEAR'].iloc[0]}")
    print(f"Training Samples: {len(train_df)}, ({train_df['CHAMP'].sum()} champion(s))")
    print(f"Validation Samples: {len(val_df)}, ({val_df['CHAMP'].sum()} champion(s))")

    feature_medians = train_df[feature_cols].median()

    X_train = train_df[feature_cols].values
    Y_train = train_df['CHAMP']
    X_val = val_df[feature_cols].values
    Y_val = val_df['CHAMP']

    # Train Model
    model, scaler = train(X_train, Y_train)
    
    # Evaluate on Validation Set
    print("\nEvaluating Logistic Regression Model:")
    evaluate(model, scaler, X_val, Y_val, team_names=val_df['TEAM'])

    # Predict for 2026 using provided 2026 training data
    df_2026 = pd.read_csv('./march+madness+data/2026training.csv')

    df_2026, feature_cols_2026 = create_features(df_2026)

    df_2026[feature_cols] = df_2026[feature_cols].fillna(feature_medians)

    X_2026 = df_2026[feature_cols].values
    X_2026_scaled = scaler.transform(X_2026)

    pred_prods_2026 = model.predict_proba(X_2026_scaled)[:, 1]
    df_2026['CHAMP PROBABILITY'] = pred_prods_2026

    df_2026_sorted = df_2026.sort_values('CHAMP PROBABILITY', ascending=False)

    print("\nTop Teams Predicted for 2026 Championship (from 2026training.csv):")
    for i, (_, row) in enumerate(df_2026_sorted.head(10).iterrows(), 1):
        print(f"{i:<6}. Team: {row['TEAM']:<25}, Predicted Probability: {row['CHAMP PROBABILITY']:.3f}")

    best_team = df_2026_sorted.iloc[0]
    print(f"\nTeam with Greatest Probability of Winning in 2026: {best_team['TEAM']} with Probability {best_team['CHAMP PROBABILITY']:.3f}")