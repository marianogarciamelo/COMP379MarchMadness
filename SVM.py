#!/usr/bin/env python3
"""
Full Tournament SVM using only RANK-BASED features.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from itertools import combinations
from collections import Counter

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import accuracy_score, f1_score

# ===================== CONFIG =====================

CSV_PATH = "./combined_features.csv"

COL_YEAR = "YEAR"
COL_TEAM = "TEAM"
COL_ROUND = "ROUND"

RANK_FEATURES = [
    "SEED",
    "TR_RANK",
    "SOS_RANK",
    "LUCK_RANK",
    "FTE_POWER_RATING_RANK",
    "KP_PRE_PRESEASON_KADJ_EM",
    "KP_PRE_PRESEASON_KADJ_O",
    "KP_PRE_PRESEASON_KADJ_D",
    "KP_PRE_PRESEASON_KADJ_T",
    "HC_EASY_DRAW",
    "HC_TOUGH_DRAW",
    "HC_DARK_HORSE",
    "HC_UPSET_ALERT",
    "HC_CINDERELLA",
]


# ROUND depth mapping: larger = deeper in tournament
ROUND_DEPTH = {
    68: 0,   # play-in
    64: 1,
    32: 2,
    16: 3,
    8:  4,
    4:  5,
    2:  6,
    1:  7,   # champion
}

ROLL_YEARS = 4                 # rolling training window
MAX_FEATURES_AFTER_L1 = 7      # max features kept after L1 LR
MC_SIMS = 800                  # Monte Carlo bracket simulations per year

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===================== UTILS =====================

def round_depth(r):
    """Convert ROUND value to depth (larger = deeper)."""
    try:
        r_int = int(r)
    except Exception:
        return 0
    return ROUND_DEPTH.get(r_int, 0)


def add_year_normalized_features(df, feature_cols):
    """
    For each YEAR and each feature, create a z-scored version:
        col_ZN = (col - mean_year) / std_year
    This makes the model learn *within-year* relative strength.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        new_col = col + "_ZN"

        zs = []
        for year, g in df.groupby(COL_YEAR):
            vals = pd.to_numeric(g[col], errors="coerce")
            mu = vals.mean()
            sigma = vals.std()
            if sigma == 0 or np.isnan(sigma):
                z = (vals - mu) * 0.0
            else:
                z = (vals - mu) / sigma
            zs.append(z)

        df[new_col] = pd.concat(zs).sort_index()

    return df


def build_pairwise_for_years(df, feature_cols, years):
    """
    Build pairwise training examples for the given list of years.
    For each YEAR and each pair of teams (a,b) in that year:
      - let da, db = depth(a), depth(b)
      - if da == db: skip (no label info)
      - else create:
            x = feats(a) - feats(b), y = 1 if da > db else 0
            x' = feats(b) - feats(a), y' = 1 - y
    Returns:
      X_raw: numpy array (n_samples, n_features)
      y:     numpy array (n_samples,)
    """
    rows_X = []
    rows_y = []

    for year in years:
        sub = df[df[COL_YEAR] == year]
        if len(sub) < 2:
            continue

        records = list(sub.to_dict("records"))
        for a, b in combinations(records, 2):
            da = round_depth(a.get(COL_ROUND, 0))
            db = round_depth(b.get(COL_ROUND, 0))
            if da == db:
                continue

            va = np.array([a.get(c, np.nan) for c in feature_cols], dtype=float)
            vb = np.array([b.get(c, np.nan) for c in feature_cols], dtype=float)
            diff_ab = va - vb
            diff_ba = -diff_ab

            if da > db:
                # A goes deeper
                rows_X.append(diff_ab)
                rows_y.append(1)
                rows_X.append(diff_ba)
                rows_y.append(0)
            else:
                # B goes deeper
                rows_X.append(diff_ba)
                rows_y.append(1)
                rows_X.append(diff_ab)
                rows_y.append(0)

    if not rows_X:
        return np.empty((0, len(feature_cols))), np.array([], dtype=int)

    X_raw = np.vstack(rows_X)
    y = np.array(rows_y, dtype=int)
    return X_raw, y


def select_features_l1_logreg(X_raw, y, feature_names):
    """
    Use L1-penalized Logistic Regression to select important features.
    Also print **all** coefficients, even if zero.
    """
    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    lr = LogisticRegression(
        penalty="l1",
        C=0.5,
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE
    )
    lr.fit(X_scaled, y)

    coefs = lr.coef_[0]
    abs_coefs = np.abs(coefs)

    print("\n--- L1 Logistic Regression: All Feature Weights ---")
    for name, w, aw in sorted(zip(feature_names, coefs, abs_coefs), key=lambda x: -x[2]):
        print(f"{name:35} weight={w: .6f}   |coef|={aw:.6f}")
    print("")

    if np.all(abs_coefs == 0):
        idx_sorted = np.arange(len(feature_names))
    else:
        idx_sorted = np.argsort(-abs_coefs)

    k = min(MAX_FEATURES_AFTER_L1, len(feature_names))
    selected_idx = idx_sorted[:k]
    selected_names = [feature_names[i] for i in selected_idx]

    print("  Selected features via L1 LR:", selected_names)
    return selected_names, selected_idx





    if np.all(abs_coefs == 0):
        idx_sorted = np.arange(len(feature_names))
    else:
        idx_sorted = np.argsort(-abs_coefs)

    k = min(MAX_FEATURES_AFTER_L1, len(feature_names))
    selected_idx = idx_sorted[:k]
    selected_names = [feature_names[i] for i in selected_idx]

    print("  Selected features via L1 LR:", selected_names)
    return selected_names, selected_idx


def train_svm_with_gridsearch(X_train, y_train, C_list, gamma_list):
    """
    Train SVM with RBF kernel using grid search over C and gamma.
    Returns:
      best_model, best_C, best_gamma, best_cv_score
    """
    param_grid = {
        "C": C_list,
        "gamma": gamma_list,
        "kernel": ["rbf"],
    }

    svc = SVC(
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_C = grid.best_params_["C"]
    best_gamma = grid.best_params_["gamma"]
    best_cv = grid.best_score_

    return best_model, best_C, best_gamma, best_cv


def build_prob_matrix(df_year, zn_features, selected_idx, imputer_svm, scaler_svm, svm_model):
    """
    Build pairwise probability matrix P[i,j] = P(team i beats team j)
    for the given year's teams.

    Steps:
      - filter to rows with ROUND > 0 and non-null SEED
      - sort by SEED ascending, then TR_RANK ascending (lower better)
      - keep top 64 teams (or fewer if less data)
      - extract year-normalized feature matrix and restrict to selected_idx
      - apply imputer + scaler
      - for each pair (i,j), compute probability via SVM on (feat_i - feat_j)
    Returns:
      prob:  (n,n) array
      teams: list of team names
    """
    df_y = df_year.copy()
    df_y = df_y[df_y[COL_ROUND] > 0]

    if "SEED" not in df_y.columns:
        return None, []

    df_y = df_y[~df_y["SEED"].isna()]
    if df_y.empty:
        return None, []

    # sort by SEED, then TR_RANK (lower is better)
    sort_cols = ["SEED"]
    ascending = [True]
    if "TR_RANK" in df_y.columns:
        sort_cols.append("TR_RANK")
        ascending.append(True)

    df_y = df_y.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    if len(df_y) > 64:
        df_y = df_y.iloc[:64].copy()

    teams = df_y[COL_TEAM].tolist()
    n = len(teams)
    if n < 2:
        return None, teams

    all_feats = df_y[zn_features].to_numpy(dtype=float)
    feats_sel = all_feats[:, selected_idx]

    feats_imp = imputer_svm.transform(feats_sel)
    feats_scaled = scaler_svm.transform(feats_imp)

    prob = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                prob[i, j] = 0.5
            else:
                diff = (feats_scaled[i] - feats_scaled[j]).reshape(1, -1)
                p = svm_model.predict_proba(diff)[0, 1]
                prob[i, j] = float(p)

    return prob, teams


def build_seedwise_bracket_order(n):
    """
    Build a simple seed-wise pairing:
      indices [0..n-1] are assumed sorted best-to-worst.
      First round: (0 vs n-1), (1 vs n-2), (2 vs n-3), ...
    Returns:
      list of indices in the order of their first-round appearance.
    """
    order = []
    left = 0
    right = n - 1
    while left < right:
        order.append(left)
        order.append(right)
        left += 1
        right -= 1
    if left == right:
        order.append(left)
    return order


def simulate_bracket(prob, bracket_order, rng):
    """
    Simulate a single-elimination bracket given:
      prob: (n,n) matrix, prob[i,j] = P(i beats j)
      bracket_order: list of team indices in first-round appearance order.
    Returns:
      champion index (0..n-1)
    """
    cur = bracket_order[:]
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            if i + 1 >= len(cur):
                nxt.append(cur[i])
                continue
            a = cur[i]
            b = cur[i + 1]
            p = prob[a, b]
            winner = a if rng.random() < p else b
            nxt.append(winner)
        cur = nxt
    return cur[0]


def monte_carlo_champ_probs(prob, teams, n_sims=MC_SIMS, seed=RANDOM_STATE):
    """
    Run Monte Carlo bracket simulations and return champion probabilities.
    Uses seed-wise bracket ordering.
    """
    rng = np.random.RandomState(seed)
    n = len(teams)
    bracket_order = build_seedwise_bracket_order(n)

    counts = Counter()
    for _ in range(n_sims):
        champ_idx = simulate_bracket(prob, bracket_order, rng)
        champ_team = teams[champ_idx]
        counts[champ_team] += 1

    for t in list(counts.keys()):
        counts[t] = counts[t] / n_sims

    return counts


# ===================== MAIN PIPELINE =====================

def main():
    global results

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.upper() for c in df.columns]

    # ensure required columns
    for c in [COL_YEAR, COL_TEAM, COL_ROUND]:
        if c not in df.columns:
            raise RuntimeError(f"Required column {c} not found in CSV.")

    # keep only rank-like features that exist
    feature_cols_available = [c for c in RANK_FEATURES if c in df.columns]

    print("Using rank feature columns:", feature_cols_available)

    # make sure features are numeric
    for col in feature_cols_available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows with missing YEAR or ROUND
    df = df.dropna(subset=[COL_YEAR, COL_ROUND])
    df[COL_YEAR] = df[COL_YEAR].astype(int)

    # add year-normalized ZN features
    df = add_year_normalized_features(df, feature_cols_available)
    zn_features = [c + "_ZN" for c in feature_cols_available]

    all_years = sorted(df[COL_YEAR].unique())
    print("Available years:", all_years)

    # Target years 2019-2024 (skip if not present)
    desired_target_years = [2019, 2020, 2021, 2022, 2023, 2024]
    target_years = [y for y in desired_target_years if y in all_years]
    print("\nWe will try to predict champions for years:", target_years)

    if not target_years:
        print("No target years found. Exiting.")
        return

    global_best_C = None
    global_best_gamma = None

    summary = []

    for ty in target_years:
        idx = all_years.index(ty)
        if idx < ROLL_YEARS:
            print(f"\nSkipping {ty}: not enough previous years for a {ROLL_YEARS}-year window.")
            continue

        train_years = all_years[idx - ROLL_YEARS: idx]

        print("\n" + "=" * 60)
        print(f"=== Evaluating year {ty} (Rank-only SVM) ===")
        print(f"  Training years: {train_years}")

        # 1) Build pairwise training data
        X_raw, y = build_pairwise_for_years(df, zn_features, train_years)
        if X_raw.shape[0] == 0:
            print("  No training pairs built; skipping this year.")
            continue

        print(f"  Training pairs: {X_raw.shape[0]}  Features per pair: {X_raw.shape[1]}")

        # 2) Feature selection via L1 Logistic Regression
        feature_names = zn_features
        selected_names, selected_idx = select_features_l1_logreg(X_raw, y, feature_names)

        # 3) Imputer + scaler for SVM on selected features
        imputer_svm = SimpleImputer(strategy="mean")
        X_imp = imputer_svm.fit_transform(X_raw[:, selected_idx])

        scaler_svm = StandardScaler()
        X_scaled = scaler_svm.fit_transform(X_imp)

        # 4) Train/dev split
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE
        )

        # 5) Train SVM
        if global_best_C is None or global_best_gamma is None:
            print("  Running SVM grid-search for this year...")
            C_list = [3, 10, 30, 100]
            gamma_list = [0.003, 0.01, 0.03]

            svm_model, best_C, best_gamma, best_cv = train_svm_with_gridsearch(
                X_train, y_train, C_list, gamma_list
            )
            global_best_C = best_C
            global_best_gamma = best_gamma
            print(f"  Grid search best C={best_C}, gamma={best_gamma}, cv_score={best_cv:.3f}")
        else:
            print(f"  Reusing global best C={global_best_C}, gamma={global_best_gamma}")
            svm_model = SVC(
                kernel="rbf",
                C=global_best_C,
                gamma=global_best_gamma,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
            svm_model.fit(X_train, y_train)

        # 6) Dev metrics
        y_dev_pred = svm_model.predict(X_dev)
        dev_acc = accuracy_score(y_dev, y_dev_pred)
        dev_f1 = f1_score(y_dev, y_dev_pred)
        print(f"  Dev accuracy: {dev_acc:.3f}, Dev F1: {dev_f1:.3f}")

        # 7) Build probability matrix for this year's field
        df_year = df[df[COL_YEAR] == ty].copy()
        if df_year.empty:
            print(f"  No teams for year {ty}; skipping simulation.")
            continue

        prob_mat, teams = build_prob_matrix(
            df_year,
            zn_features,
            selected_idx,
            imputer_svm,
            scaler_svm,
            svm_model
        )

        if prob_mat is None or len(teams) < 2:
            print(f"  Not enough teams in year {ty} to simulate.")
            continue

        print(f"  Running Monte Carlo bracket simulation ({MC_SIMS} sims)...")
        champ_probs = monte_carlo_champ_probs(prob_mat, teams, n_sims=MC_SIMS, seed=RANDOM_STATE)

        # Predicted champion
        pred_team, pred_p = max(champ_probs.items(), key=lambda kv: kv[1])

        # Actual champion = deepest ROUND in that year
        df_year["DEPTH"] = df_year[COL_ROUND].apply(round_depth)
        actual_row = df_year.loc[df_year["DEPTH"].idxmax()]
        actual_team = actual_row[COL_TEAM]

        # Rank of actual champ among all teams
        sorted_probs = sorted(champ_probs.items(), key=lambda kv: kv[1], reverse=True)
        all_names = [t for t, _ in sorted_probs]
        if actual_team in all_names:
            rank_all = all_names.index(actual_team) + 1
        else:
            rank_all = None

        # Rank within top 4
        top4 = all_names[:4]
        if actual_team in top4:
            rank_top4 = top4.index(actual_team) + 1
        else:
            rank_top4 = None

        correct = (pred_team == actual_team)

        print(f"  Predicted champion: {pred_team} (p={pred_p:.3f})")
        print(f"  Actual champion:    {actual_team}")
        print(f"  Correct champion prediction? {correct}")
        if rank_all is not None:
            print(f"  Actual champ rank among all teams: {rank_all}/{len(all_names)}")
        else:
            print("  Actual champ not found in Monte Carlo ranking set.")
        if rank_top4 is not None:
            print(f"  Actual champ rank within top 4: {rank_top4}/4")
        else:
            print("  Actual champ not in top 4 predicted.")

        # Record for summary + external use
        results.append((
            ty,
            pred_team,
            actual_team,
            f"{rank_all}/{len(all_names)}" if rank_all is not None else None,
            dev_acc,
            dev_f1,
            global_best_C,
            global_best_gamma
        ))

        summary.append({
            "year": ty,
            "pred": pred_team,
            "pred_p": pred_p,
            "actual": actual_team,
            "correct": correct,
            "rank_all": rank_all,
            "n_teams": len(all_names),
            "rank_top4": rank_top4,
            "dev_acc": dev_acc,
            "dev_f1": dev_f1,
            "C": global_best_C,
            "gamma": global_best_gamma,
            "selected_features": selected_names,
        })

    # ===================== OVERALL SUMMARY =====================
    print("\n" + "=" * 60)
    print("SUMMARY (Rank-only SVM):")
    if not summary:
        print("  No years were successfully evaluated.")
        return

    correct_count = sum(1 for s in summary if s["correct"])
    print(f"  Correct champion predictions: {correct_count}/{len(summary)}")

    for s in summary:
        rank_all_str = (
            f"{s['rank_all']}/{s['n_teams']}" if s['rank_all'] is not None else "N/A"
        )
        rank_top4_str = (
            f"{s['rank_top4']}/4" if s['rank_top4'] is not None else "N/A"
        )
        print(
            f" {s['year']}: pred={s['pred']} (p={s['pred_p']:.3f}), "
            f"actual={s['actual']}, rank_all={rank_all_str}, "
            f"rank_top4={rank_top4_str}, dev_acc={s['dev_acc']:.3f}, "
            f"dev_f1={s['dev_f1']:.3f}"
        )

# For comparison scripts
def get_results():
    return results

if __name__ == "__main__":
    main()
