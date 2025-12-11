# Neural Network Bracket Simulator

This folder holds a lightweight neural network to train a pairwise game predictor using scikit-learn MLP and simulate an NCAA tournament bracket round-by-round.

## Files
- `data_prep.py`: loads `combined_features.csv` and `march+madness+data/Tournament Matchups.csv` and then builds the pairwise training rows, and standardizes matchup vectors.
- `model.py`: defines `GamePredictor`, a scikit-learn `MLPClassifier` (Multi-layer Perceptron classifier) with a `StandardScaler`.
- `simulator.py`: trains (or loads) the model and runs Monte Carlo simulations for a user-selected year.

## Quickstart
1) Install deps (inside your venv):
```bash
pip install -r requirements.txt
```
2) Run a simulation for a given year (trains fresh unless a saved model exists):
```bash
python -m nathan_neural_net.cli simulate --year 2024 --sims 500 --verbose
```

`simulate_year` excludes the requested `year` from training to avoid leaking actual results. Set `simulations` to control Monte Carlo runs and `model_dir` to cache/reuse the trained network.

## Quick accuracy check on a past year
This trains on all other years and evaluates on the chosen year using the true scores from `Tournament Matchups.csv`:
```bash
python -m nathan_neural_net.cli evaluate --year 2024 --verbose
# add --include-target-in-training to allow leakage for a full-data fit
```
Metrics reported: Accuracy, Precision, F1, Log loss, AUC, plus counts of evaluated vs. skipped games.

## Hyperparameter tuning (grid search)
Try multiple hidden-layer configs, learning rates, dropouts, batches, and epochs:
```bash
python -m nathan_neural_net.cli tune --year 2024 --verbose \
  --hidden-grid "128,64" "256,128,64" \
  --lr-grid 0.001 0.0005 \
  --dropout-grid 0.0 0.1 \
  --batch-grid 32 64 \
  --epoch-grid 50 100 \
  --top-k 5 \
  --model-dir /tmp/nn_best   # optional: save best-by-logloss model here
```

## More "intense" model
If you want to build a model with more neurons and train it for longer you can run something like the code below. You can also checkout `nathan_neural_net/cli.py` for all the flags and options
```
python -m nathan_neural_net.cli evaluate --year 2024 \
  --hidden-units 512 256 128 64 \
  --lr 0.0005 \
  --batch-size 32 \
  --epochs 200 \
  --verbose
```

## Notes
- Training data is built from historical tournament scores; games missing scores are skipped.
- Features are symmetric: for each game we include both `(team_a - team_b)` and `(team_b - team_a)` rows to balance the classes.
- The base `ROUND` column from `combined_features.csv` is dropped to avoid leaking actual tournament outcomes into the model.
- If a team's features are missing for the requested year, the simulator will raise an explicit error so you can update `combined_features.csv`.
