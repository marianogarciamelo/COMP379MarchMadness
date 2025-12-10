import argparse
from pathlib import Path
from typing import List, Optional

from .simulator import EvaluationResult, evaluate_year, simulate_year


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
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

    return parser.parse_args()


def run_simulate(args: argparse.Namespace) -> None:
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
    print(f"Year: {res.year}")
    print(f"Games evaluated: {res.games}, Missing feature rows: {res.missing_games}")
    print(
        f"Accuracy: {res.accuracy:.3f}, Precision: {res.precision:.3f}, "
        f"F1: {res.f1:.3f}, Log loss: {res.log_loss:.3f}, AUC: {res.auc}"
    )


def run_evaluate(args: argparse.Namespace) -> None:
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


def main() -> None:
    args = parse_args()
    if args.command == "simulate":
        run_simulate(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
