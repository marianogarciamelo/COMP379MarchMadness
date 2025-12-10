# Neural Network Bracket Simulator

This folder holds a lightweight pipeline to train a pairwise game predictor (scikit-learn MLP) and simulate an NCAA tournament bracket round-by-round.

## Files
- `data_prep.py`: loads `combined_features.csv` and `march+madness+data/Tournament Matchups.csv`, builds pairwise training rows, and standardizes matchup vectors.
- `model.py`: defines `GamePredictor`, a scikit-learn `MLPClassifier` with a `StandardScaler`.
- `simulator.py`: trains (or loads) the model and runs Monte Carlo simulations for a user-selected year.

## Quickstart
1) Install deps (inside your virtualenv):
```bash
pip install -r requirements.txt
```
2) Run a simulation for a given year (trains fresh unless a saved model exists):
```bash
python -m neural_net.cli simulate --year 2024 --sims 500 --model-dir neural_net/saved_model
# add --verbose to see training/loading progress
```

`simulate_year` excludes the requested `year` from training to avoid leaking actual results. Set `simulations` to control Monte Carlo runs and `model_dir` to cache/reuse the trained network.

## Quick accuracy check on a past year
This trains on all other years and evaluates on the chosen year using the true scores from `Tournament Matchups.csv`:
```bash
python -m neural_net.cli evaluate --year 2024
# add --include-target-in-training to allow leakage for a full-data fit
```
Metrics reported: Accuracy, Precision, F1, Log loss, AUC, plus counts of evaluated vs. skipped games.

## Notes
- Training data is built from historical tournament scores; games missing scores are skipped.
- Features are symmetric: for each game we include both `(team_a - team_b)` and `(team_b - team_a)` rows to balance the classes.
- The base `ROUND` column from `combined_features.csv` is dropped to avoid leaking actual tournament outcomes into the model.
- If a team's features are missing for the requested year, the simulator will raise an explicit error so you can update `combined_features.csv`.
