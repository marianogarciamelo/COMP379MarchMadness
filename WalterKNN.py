import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_builder import materialize_dataset, OUTPUT_PATH

def load_data():
    df = pd.read_parquet(OUTPUT_PATH)
    return df

def prep_data(df):
    df = df.copy()
    df = df.dropna(subset=["ROUND"])
    df["ROUND"] = df["ROUND"].astype(int)
    round_mapping = {
        68: 0,# for the play in games
        64: 1,
        32: 2,
        16: 3,
        8: 4,
        4: 5,
        2: 6,
        1: 7 # 1 will be the champion
    }
    df["ROUND"] = df["ROUND"].map(round_mapping)
    df = df.dropna(subset=["ROUND"])
    df["ROUND"] = df["ROUND"].astype(int)
    numeric_df= df.select_dtypes(include =[np.number])

    remove_columns = ["YEAR", "TEAM_NO", "ROUND"]
    feature_columns =[col for col in numeric_df.columns if col not in remove_columns]

    X = numeric_df[feature_columns]
    y = numeric_df["ROUND"]

    print(f"Number of features: {X.shape[1]}")
    print("ROUND distribution:")
    print(y.value_counts().sort_index())
    return X, y, feature_columns

def build_knn_model(X, y, k = 7):
    imputer = SimpleImputer(strategy = "mean")
    X_imputed = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size= 0.25, random_state= 42, stratify = y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors = k))
    ])
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    print("\nKNN MODEL PERFORMANCE:")
    print("Accuracy:", round(accuracy_score(y_test, prediction), 4))
    print("F1 Score:", round(f1_score(y_test, prediction, average= "weighted"), 4 ))

    print("\n Classification Report:")
    print(classification_report(y_test, prediction))

    return model, imputer

def predict_round(model, df, imputer, feature_columns, year):
    df_year = df[df["YEAR"]== year].copy()
    numeric_df = df_year.select_dtypes(include=[np.number])
    X_year = numeric_df[feature_columns]

    X_year = imputer.transform(X_year)

    #get the probabilities for each round
    prob_matrix = model.predict_proba(X_year)

    expected_round= (prob_matrix * np.arange(prob_matrix.shape[1])).sum(axis=1)

    df_year["EXPECTED_ROUND"] = expected_round
    df_year["PREDICTED_ROUND"] = model.predict(X_year)

    return df_year.sort_values("EXPECTED_ROUND", ascending = False)[
        ["TEAM", "EXPECTED_ROUND", "PREDICTED_ROUND"]]

if __name__ == "__main__":
    materialize_dataset()
    df = load_data()
    X, y, feature_columns = prep_data(df)

    model, imputer = build_knn_model(X, y, k =7)

    print("\n Predictions for 2024:")
    print(predict_round(model, df, imputer, feature_columns, 2024).head(10))

    print("\n Predictions for 2023: ")
    print(predict_round(model, df, imputer, feature_columns, 2023).head(10))