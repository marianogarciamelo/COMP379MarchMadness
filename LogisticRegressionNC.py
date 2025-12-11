# GOAL: Identify a single team that has the greatest probability of winning in 2026

# String Values: Classifier (Team Name)
# Int Values: Year, Team_#, Seed, Elimination Round, TR_Rank, SOS_Rank, Luck_Rank
# Double Values: TR_Rating, SOS_Rating, Luck_Rating (CAN BE NEGATIVE)


# L1 Regularization will solve redundant features


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler



# URL
data = './combined_features.csv'

# Load Data
df = pd.read_csv(data)

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

    # Add FTE or KP features if available (2012+)
    if 'FTE_POWER_RATING' in df.columns:
        df['FTE_POWER_RATING'] = df['FTE_POWER_RATING'].fillna(df['FTE_POWER_RATING'].median())
        feature_cols.append('FTE_POWER_RATING') # Add FTE Power Rating if exists
    
    if 'KP_PRE_PRESEASON_KADJ_EM' in df.columns:
        df['KP_PRE_PRESEASON_KADJ_EM'] = df['KP_PRE_PRESEASON_KADJ_EM'].fillna(df['KP_PRE_PRESEASON_KADJ_EM'].median())
        feature_cols.append('KP_PRE_PRESEASON_KADJ_EM') # Add KP Preseason KADJ if exists

    # Weighing Rank Features 
    df['Rank_SOS_Interaction'] = df['TR_RANK'] * df['SOS_RANK']
    feature_cols.append('Rank_SOS_Interaction')

    # Rating Advantages
    df['Rating_vs_SOS'] = df['TR_RATING'] - (17 - df['SEED'])
    feature_cols.append('Rating_vs_SOS')

    df['Seed_Squared'] = df['SEED'] ** 2
    feature_cols.append('Seed_Squared')

    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df, feature_cols



def prepare_data(df, target_year=None, exclude_recent=False):
    # Prepare training and test date with 2026 as future data
    
    # Prepare data for modeling
    df, feature_cols = create_features(df)

    # Define target variable
    df['CHAMP'] = (df['ROUND'] == 1).astype(int)

    if target_year:
        #Predict specific year

        # Training
        train_mask = (df['YEAR'] < target_year) & (df['ROUND'] > 0)
        pred_mask = (df['YEAR'] == target_year) & (df['ROUND'] == 0)

        train_df = df[train_mask].copy()
        pred_df = df[pred_mask].copy()

        return train_df, pred_df, feature_cols
    
    else:
        # Use all past data for training, future data for prediction
        historical_df = df[df['ROUND'] > 0].copy()

        if exclude_recent:
            # Exclude most recent year from training
            max_year = historical_df['YEAR'].max()
            train_df = historical_df[historical_df['YEAR'] < max_year]
            val_df = historical_df[historical_df['YEAR'] == max_year]
            return train_df, val_df, feature_cols
        else:
            return historical_df, val_df, feature_cols



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

def predict(train_df, pred_df, feature_cols):
    # Train on all historical data and predict for 2026

    X_train = train_df[feature_cols].values
    Y_train = train_df['CHAMP']

    print("Training on all historical data...")
    model, scaler = train(X_train, Y_train)

    X_pred = pred_df[feature_cols].values
    X_pred_scaled = scaler.transform(X_pred)

    pred_prods = model.predict_proba(X_pred_scaled)[:, 1]
    pred_df = pred_df.copy()
    pred_df['CHAMP PROBABILITY'] = pred_prods

    pred_df_sorted = pred_df.sort_values('CHAMP PROBABILITY', ascending=False)


    print("Top Teams Predicted for 2026 Championship:")
    for i, (_, row) in enumerate(pred_df_sorted.head(10).iterrows(), 1):
        print(f"{i:<6}. Team: {row['TEAM']:<25}, Predicted Probability: {row['CHAMP PROBABILITY']:.3f}")
    
    return pred_df_sorted, model, scaler

if __name__ == "__main__":
    # Load Data
    df = pd.read_csv(data)

    # Prepare Data for 2026 Prediction
    train_df, val_df, feature_cols = prepare_data(df, exclude_recent=True)
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