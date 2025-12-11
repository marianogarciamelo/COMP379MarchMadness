""" File to do a simple grid search over MLP hyperparameters. """

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .data_prep import _iter_games, build_training_dataset, matchup_vector
from .model import GamePredictor


@dataclass
class TrialResult:
    hidden_units: Sequence[int]
    lr: float
    dropout: float
    batch_size: int
    epochs: int
    accuracy: float
    precision: float
    f1: float
    log_loss: float
    auc: float | None


def _evaluate_with_dataset(dataset, predictor: GamePredictor, target_year: int):
    """Evaluate a trained predictor on the target year using prebuilt dataset/matchups."""
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

    from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, roc_auc_score

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

    return accuracy, precision, f1, logloss, auc


def grid_search(
    target_year: int,
    hidden_grid: Iterable[Sequence[int]],
    lrs: Iterable[float],
    dropouts: Iterable[float],
    batch_sizes: Iterable[int],
    epochs_list: Iterable[int],
    save_dir: Path | None = None,
    verbose: bool = False,
    top_k: int = 5,
) -> List[TrialResult]:
    """Run a simple grid search over MLP hyperparameters and return the top results."""
    dataset = build_training_dataset(exclude_year=target_year)
    results: List[TrialResult] = []
    best_loss = float("inf")

    for hidden, lr, dropout, batch_size, epochs in product(
        hidden_grid, lrs, dropouts, batch_sizes, epochs_list
    ):
        if verbose:
            print(
                f"Trial hidden={hidden}, lr={lr}, dropout={dropout}, "
                f"batch_size={batch_size}, epochs={epochs}"
            )
        predictor = GamePredictor(
            hidden_units=hidden,
            learning_rate=lr,
            batch_size=batch_size,
            epochs=epochs,
            dropout=dropout,
        )
        predictor.fit(dataset.X, dataset.y, dataset.feature_names)
        accuracy, precision, f1, logloss, auc = _evaluate_with_dataset(dataset, predictor, target_year)
        results.append(
            TrialResult(
                hidden_units=hidden,
                lr=lr,
                dropout=dropout,
                batch_size=batch_size,
                epochs=epochs,
                accuracy=accuracy,
                precision=precision,
                f1=f1,
                log_loss=logloss,
                auc=auc,
            )
        )

        if save_dir and logloss < best_loss:
            if verbose:
                print(f"New best log_loss={logloss:.4f}; saving model to {save_dir}")
            predictor.save(save_dir)
            best_loss = logloss

    # Sort by log loss ascending, then by accuracy descending.
    results.sort(key=lambda r: (r.log_loss, -r.accuracy))
    return results[:top_k]
