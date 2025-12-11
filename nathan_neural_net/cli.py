"""CLI for training, evaluating, and simulating with the neural net model."""

import argparse
from pathlib import Path
from typing import List, Optional

from .simulator import EvaluationResult, evaluate_year, simulate_year
from .tuner import grid_search


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    """Shared CLI flags for model hyperparameters."""
    parser.add_argument("--hidden-units", nargs="+", type=int, default=[128, 64], help="Hidden layer sizes.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=40, help="Max training epochs.")
    parser.add_argument("--verbose", action="store_true", help="Print progress details.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bracket simulations or evaluate accuracy.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim = subparsers.add_parser("simulate", help="Simulate a tournament for a given year.")
    sim.add_argument("--year", type=int, required=True, help="Year to simulate (excluded from training).")
    sim.add_argument("--sims", type=int, default=500, help="Number of Monte Carlo simulations.")
    sim.add_argument("--model-dir", type=Path, default=None, help="Directory to load/save the trained model.")
    sim.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    _add_common_train_args(sim)

    evalp = subparsers.add_parser("evaluate", help="Evaluate accuracy on a past year.")
    evalp.add_argument("--year", type=int, required=True, help="Year to evaluate using actual scores.")
    evalp.add_argument(
        "--include-target-in-training",
        action="store_true",
        help="If set, allow training on the target year (leaks that year's outcomes).",
    )
    _add_common_train_args(evalp)

    tune = subparsers.add_parser("tune", help="Grid search over hyperparameters for a given year.")
    tune.add_argument("--year", type=int, required=True, help="Year to evaluate using actual scores.")
    tune.add_argument(
        "--hidden-grid",
        nargs="+",
        default=["128,64", "256,128,64"],
        help="Hidden layer configs, comma-separated (e.g., '128,64' '256,128,64').",
    )
    tune.add_argument(
        "--lr-grid",
        nargs="+",
        type=float,
        default=[1e-3, 5e-4],
        help="Learning rates to try.",
    )
    tune.add_argument(
        "--dropout-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.1],
        help="Dropout rates to try.",
    )
    tune.add_argument(
        "--batch-grid",
        nargs="+",
        type=int,
        default=[32, 64],
        help="Batch sizes to try.",
    )
    tune.add_argument(
        "--epoch-grid",
        nargs="+",
        type=int,
        default=[50, 100],
        help="Epoch counts to try.",
    )
    tune.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="If set, save the best-performing model here (by log loss).",
    )
    tune.add_argument("--top-k", type=int, default=5, help="Number of top configs to print.")
    tune.add_argument("--verbose", action="store_true", help="Print progress details.")

    return parser.parse_args()


def _parse_hidden_grid(raw: List[str]) -> List[tuple[int, ...]]:
    """Parse comma-separated hidden-layer specs like '128,64' into tuples."""
    parsed = []
    for item in raw:
        parts = [p for p in item.replace(" ", "").split(",") if p]
        if not parts:
            continue
        parsed.append(tuple(int(p) for p in parts))
    return parsed


def run_simulate(args: argparse.Namespace) -> None:
    """Entry point for the simulate subcommand."""
    result = simulate_year(
        year=args.year,
        simulations=args.sims,
        model_dir=args.model_dir,
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        verbose=args.verbose,
    )
    print(f"Ran {result.simulations} simulations for {args.year}. Top champions:")
    for team, prob in list(result.champion_probs.items())[:10]:
        print(f"{team}: {prob:.3f}")


def _print_eval(res: EvaluationResult) -> None:
    """Format evaluation metrics for console output."""
    print(f"Year: {res.year}")
    print(f"Games evaluated: {res.games}, Missing feature rows: {res.missing_games}")
    print(
        f"Accuracy: {res.accuracy:.3f}, Precision: {res.precision:.3f}, "
        f"F1: {res.f1:.3f}, Log loss: {res.log_loss:.3f}, AUC: {res.auc}"
    )


def run_evaluate(args: argparse.Namespace) -> None:
    """Entry point for the evaluate subcommand."""
    res = evaluate_year(
        target_year=args.year,
        include_target_in_training=args.include_target_in_training,
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
    )
    _print_eval(res)


def run_tune(args: argparse.Namespace) -> None:
    """Run a lightweight grid search over hyperparameters and print top results."""
    hidden_grid = _parse_hidden_grid(args.hidden_grid)
    results = grid_search(
        target_year=args.year,
        hidden_grid=hidden_grid,
        lrs=args.lr_grid,
        dropouts=args.dropout_grid,
        batch_sizes=args.batch_grid,
        epochs_list=args.epoch_grid,
        save_dir=args.model_dir,
        verbose=args.verbose,
        top_k=args.top_k,
    )
    print(f"Top {len(results)} configs by log loss (primary) then accuracy:")
    for r in results:
        print(
            f"hidden={r.hidden_units}, lr={r.lr}, dropout={r.dropout}, "
            f"batch={r.batch_size}, epochs={r.epochs} -> "
            f"acc={r.accuracy:.3f}, prec={r.precision:.3f}, f1={r.f1:.3f}, "
            f"logloss={r.log_loss:.3f}, auc={r.auc}"
        )


def main() -> None:
    args = parse_args()
    if args.command == "simulate":
        run_simulate(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "tune":
        run_tune(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
