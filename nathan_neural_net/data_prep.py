from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_PATH = BASE_DIR / "combined_features.csv"
MATCHUPS_PATH = BASE_DIR / "march+madness+data" / "Tournament Matchups.csv"


@dataclass
class PairwiseDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    base_features: List[str]
    feature_frame: pd.DataFrame
    matchups: pd.DataFrame
    skipped_games: int


def _standardize_team(value: str) -> str:
    return str(value).strip().upper()


def load_team_features(path: Path = FEATURE_PATH) -> Tuple[pd.DataFrame, List[str]]:
    """Load the combined feature table and return it plus the usable feature columns."""
    leakage_cols = {"ROUND"}  # Post-tournament outcomes; exclude to avoid leakage.
    df = pd.read_csv(path)
    df.columns = [col.upper() for col in df.columns]
    df["TEAM"] = df["TEAM"].map(_standardize_team)
    df["YEAR"] = df["YEAR"].astype(int)

    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    feature_cols = [col for col in df.columns if col not in {"YEAR", "TEAM"}]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    return df, feature_cols


def load_matchups(path: Path = MATCHUPS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.upper() for col in df.columns]
    df["TEAM"] = df["TEAM"].map(_standardize_team)
    df["YEAR"] = df["YEAR"].astype(int)
    df["CURRENT ROUND"] = pd.to_numeric(df["CURRENT ROUND"], errors="coerce").astype(int)
    df["BY YEAR NO"] = pd.to_numeric(df["BY YEAR NO"], errors="coerce")
    df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce")
    df["SEED"] = pd.to_numeric(df["SEED"], errors="coerce")
    df = df.drop_duplicates(subset=["YEAR", "TEAM", "CURRENT ROUND"])
    return df


def _iter_games(
    matchups: pd.DataFrame,
    exclude_year: Optional[int] = None,
    require_scores: bool = False,
) -> Iterable[Tuple[int, int, pd.Series, pd.Series]]:
    df = matchups
    if exclude_year is not None:
        df = df[df["YEAR"] != exclude_year]

    for (year, round_no), group in df.groupby(["YEAR", "CURRENT ROUND"]):
        sorted_group = group.sort_values("BY YEAR NO", ascending=False)
        if len(sorted_group) % 2 != 0:
            continue
        for i in range(0, len(sorted_group), 2):
            team_a = sorted_group.iloc[i]
            team_b = sorted_group.iloc[i + 1]
            if require_scores and (pd.isna(team_a["SCORE"]) or pd.isna(team_b["SCORE"])):
                continue
            yield year, round_no, team_a, team_b


def _diff_vector(
    feature_lookup: pd.DataFrame,
    feature_cols: List[str],
    year: int,
    team_a: str,
    team_b: str,
) -> np.ndarray:
    try:
        vec_a = feature_lookup.loc[(year, team_a), feature_cols].values.astype(float)
        vec_b = feature_lookup.loc[(year, team_b), feature_cols].values.astype(float)
    except KeyError as exc:
        raise KeyError(f"Missing features for {team_a} or {team_b} in {year}") from exc
    return vec_a - vec_b


def build_training_dataset(
    exclude_year: Optional[int] = None,
    feature_path: Path = FEATURE_PATH,
    matchups_path: Path = MATCHUPS_PATH,
    augment: bool = True,
) -> PairwiseDataset:
    features, feature_cols = load_team_features(feature_path)
    matchups = load_matchups(matchups_path)
    feature_lookup = features.set_index(["YEAR", "TEAM"])

    rows: List[np.ndarray] = []
    labels: List[int] = []
    skipped = 0
    for year, round_no, team_a, team_b in _iter_games(
        matchups, exclude_year=exclude_year, require_scores=True
    ):
        try:
            diff = _diff_vector(feature_lookup, feature_cols, year, team_a["TEAM"], team_b["TEAM"])
        except KeyError:
            skipped += 1
            continue

        vector = diff
        winner = 1 if float(team_a["SCORE"]) > float(team_b["SCORE"]) else 0
        rows.append(vector)
        labels.append(winner)

        if augment:
            alt_vector = -diff
            alt_winner = 1 - winner
            rows.append(alt_vector)
            labels.append(alt_winner)

    if not rows:
        raise ValueError("No training rows were produced; check your data paths.")

    feature_names = feature_cols
    return PairwiseDataset(
        X=np.vstack(rows),
        y=np.array(labels, dtype=int),
        feature_names=feature_names,
        base_features=feature_cols,
        feature_frame=features,
        matchups=matchups,
        skipped_games=skipped,
    )


def matchup_vector(
    feature_table: pd.DataFrame,
    feature_cols: List[str],
    year: int,
    team_a: str,
    team_b: str,
    round_no: int,
    feature_lookup: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    lookup = feature_lookup if feature_lookup is not None else feature_table.set_index(["YEAR", "TEAM"])
    diff = _diff_vector(lookup, feature_cols, year, team_a, team_b)
    return diff
