from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, roc_auc_score
import time

from .data_prep import (
    PairwiseDataset,
    _iter_games,
    build_training_dataset,
    load_matchups,
    load_team_features,
    matchup_vector,
)
from .model import GamePredictor


def _round_pairs(matchups: pd.DataFrame, year: int, round_no: int) -> List[Tuple[str, str]]:
    subset = matchups[(matchups["YEAR"] == year) & (matchups["CURRENT ROUND"] == round_no)]
    ordered = subset.sort_values("BY YEAR NO", ascending=False)
    if len(ordered) % 2 != 0:
        raise ValueError(f"Round {round_no} for {year} has an odd number of teams.")
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(ordered), 2):
        team_a = ordered.iloc[i]["TEAM"]
        team_b = ordered.iloc[i + 1]["TEAM"]
        pairs.append((team_a, team_b))
    return pairs


@dataclass
class SimulationResult:
    champion_probs: Dict[str, float]
    champion_counts: Dict[str, int]
    simulations: int


@dataclass
class EvaluationResult:
    year: int
    games: int
    missing_games: int
    accuracy: float
    precision: float
    f1: float
    auc: Optional[float]
    log_loss: float


class TournamentSimulator:
    def __init__(
        self,
        predictor: GamePredictor,
        feature_frame: pd.DataFrame,
        feature_cols: Sequence[str],
        matchups: pd.DataFrame,
    ) -> None:
        self.predictor = predictor
        self.feature_frame = feature_frame
        self.feature_cols = list(feature_cols)
        self.feature_lookup = feature_frame.set_index(["YEAR", "TEAM"])
        self.matchups = matchups

    def _simulate_round(
        self, year: int, pairs: List[Tuple[str, str]], round_no: int, rng: np.random.Generator
    ) -> List[str]:
        winners: List[str] = []
        for team_a, team_b in pairs:
            vector = matchup_vector(
                self.feature_frame,
                self.feature_cols,
                year,
                team_a,
                team_b,
                round_no,
                feature_lookup=self.feature_lookup,
            )
            prob = self.predictor.predict_game_prob(vector)
            winner = team_a if rng.random() < prob else team_b
            winners.append(winner)
        return winners

    def simulate_tournament(
        self, year: int, simulations: int = 500, seed: Optional[int] = None
    ) -> SimulationResult:
        rounds = sorted(
            self.matchups[self.matchups["YEAR"] == year]["CURRENT ROUND"].unique(), reverse=True
        )
        if not rounds:
            raise ValueError(f"No matchups found for {year}.")
        current_round = rounds[0]
        pairs = _round_pairs(self.matchups, year, current_round)
        rng = np.random.default_rng(seed)
        champion_counter: Counter[str] = Counter()

        for _ in range(simulations):
            round_no = current_round
            current_pairs = pairs
            while current_pairs:
                winners = self._simulate_round(year, current_pairs, round_no, rng)
                if len(winners) == 1:
                    champion_counter[winners[0]] += 1
                    break
                # Preserve bracket order when pairing winners
                next_pairs = []
                for i in range(0, len(winners), 2):
                    next_pairs.append((winners[i], winners[i + 1]))
                round_no = max(2, round_no // 2)
                current_pairs = next_pairs

        total = sum(champion_counter.values())
        probs = {team: count / total for team, count in champion_counter.items()}
        return SimulationResult(
            champion_probs=dict(sorted(probs.items(), key=lambda item: item[1], reverse=True)),
            champion_counts=dict(champion_counter),
            simulations=simulations,
        )


def train_predictor(
    sim_year: Optional[int] = None,
    hidden_units: Sequence[int] = (128, 64),
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 40,
    verbose: bool = False,
) -> Tuple[GamePredictor, PairwiseDataset]:
    dataset = build_training_dataset(exclude_year=sim_year)
    if verbose:
        print(
            f"Built training set: {dataset.X.shape[0]} rows, "
            f"{len(dataset.base_features)} base features, skipped {dataset.skipped_games} games"
        )

    predictor = GamePredictor(
        hidden_units=hidden_units,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        dropout=dropout,
    )

    if verbose:
        print(f"Training model (epochs={epochs}, hidden_layers={hidden_units}, dropout={dropout})...")
        start = time.time()
    predictor.fit(dataset.X, dataset.y, dataset.feature_names)
    if verbose:
        print(f"Training complete in {time.time() - start:.1f}s")
    return predictor, dataset


def evaluate_year(
    target_year: int,
    include_target_in_training: bool = False,
    verbose: bool = False,
    **train_kwargs,
) -> EvaluationResult:
    predictor, dataset = train_predictor(
        sim_year=None if include_target_in_training else target_year,
        verbose=verbose,
        **train_kwargs,
    )
    feature_lookup = dataset.feature_frame.set_index(["YEAR", "TEAM"])

    rows: List[np.ndarray] = []
    labels: List[int] = []
    missing = 0
    for year, round_no, team_a, team_b in _iter_games(
        dataset.matchups, exclude_year=None, require_scores=True
    ):
        if year != target_year:
            continue
        try:
            vec = matchup_vector(
                dataset.feature_frame,
                dataset.base_features,
                year,
                team_a["TEAM"],
                team_b["TEAM"],
                round_no,
                feature_lookup=feature_lookup,
            )
        except KeyError:
            missing += 1
            continue
        winner = 1 if float(team_a["SCORE"]) > float(team_b["SCORE"]) else 0
        rows.append(vec)
        labels.append(winner)

    if not rows:
        raise ValueError(f"No scored games found for {target_year}.")

    X_test = np.vstack(rows)
    y_true = np.array(labels, dtype=int)
    probs = predictor.predict_proba(X_test)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    preds = (probs >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_true, preds))
    logloss = float(log_loss(y_true, probs))
    precision = float(precision_score(y_true, preds))
    f1 = float(f1_score(y_true, preds))
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = None

    return EvaluationResult(
        year=target_year,
        games=len(y_true),
        missing_games=missing,
        accuracy=accuracy,
        precision=precision,
        f1=f1,
        auc=auc,
        log_loss=logloss,
    )


def simulate_year(
    year: int,
    simulations: int = 500,
    model_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    **train_kwargs,
) -> SimulationResult:
    predictor = None
    features: Optional[pd.DataFrame] = None
    feature_cols: Optional[List[str]] = None
    matchups: Optional[pd.DataFrame] = None

    if model_dir and Path(model_dir).exists():
        if verbose:
            print(f"Loading cached model from {model_dir} ...")
        predictor = GamePredictor.load(model_dir)
        features, feature_cols = load_team_features()
        matchups = load_matchups()

    if predictor is None:
        predictor, dataset = train_predictor(
            sim_year=year, verbose=verbose, **train_kwargs
        )
        features, feature_cols = dataset.feature_frame, dataset.base_features
        matchups = dataset.matchups
        if model_dir:
            if verbose:
                print(f"Saving trained model to {model_dir} ...")
            predictor.save(model_dir)

    simulator = TournamentSimulator(
        predictor=predictor,
        feature_frame=features,
        feature_cols=feature_cols,
        matchups=matchups,
    )
    if verbose:
        print(f"Running {simulations} bracket simulations for {year} ...")
    return simulator.simulate_tournament(year=year, simulations=simulations, seed=seed)


if __name__ == "__main__":
    result = simulate_year(year=2024, simulations=200)
    print("Top champion probabilities:")
    for team, prob in list(result.champion_probs.items())[:10]:
        print(f"{team}: {prob:.3f}")
