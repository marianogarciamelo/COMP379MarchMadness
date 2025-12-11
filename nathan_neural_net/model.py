from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


class GamePredictor:
    """Sklearn-based MLP game predictor. 
    Not using TensorFlow/Keras because it was giving me so many issues.

    The pipeline is simple:
    - Standardize features with StandardScaler (fit on training data only).
    - Train an MLPClassifier with class-balancing via sample weights.
    - Provide save/load helpers so the trained scaler/model can be reused.
    """

    def __init__(
        self,
        hidden_units: Sequence[int] = (128, 64),
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 40,
        dropout: float = 0.0,
    ) -> None:
        self.hidden_units = tuple(hidden_units)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout  # Kept for interface symmetry; not used by MLPClassifier.

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[MLPClassifier] = None
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Iterable[str]) -> None:
        # Persist feature ordering for later predictions/saving.
        self.feature_names = list(feature_names)
        # Normalize inputs so each feature has mean 0, std 1.
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Compute balanced weights to offset any class imbalance (favorites vs underdogs).
        class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
        cw = {0: class_weights[0], 1: class_weights[1]}
        sample_weight = np.array([cw[int(label)] for label in y], dtype=float)

        print(feature_names)
        # MLP with early stopping on a held-out validation split.
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_units,
            activation="relu",
            learning_rate_init=self.learning_rate,
            batch_size=self.batch_size,
            max_iter=self.epochs,
            early_stopping=True,
            n_iter_no_change=5,
            validation_fraction=0.2,
            verbose=False,
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model has not been trained or loaded.")
        # Ensure shape is (n_samples, n_features)
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        scaled = self.scaler.transform(arr)
        probs = self.model.predict_proba(scaled)[:, 1]
        return probs

    def predict_game_prob(self, vector: np.ndarray) -> float:
        return float(self.predict_proba(vector)[0])

    def save(self, directory: Path) -> None:
        import joblib

        if self.model is None or self.scaler is None:
            raise RuntimeError("Train the model before saving.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "sklearn_model.joblib"
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "hidden_units": self.hidden_units,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, directory: Path) -> "GamePredictor":
        import joblib

        directory = Path(directory)
        path = directory / "sklearn_model.joblib"
        if not path.exists():
            raise FileNotFoundError("Expected sklearn_model.joblib is missing.")
        payload = joblib.load(path)
        instance = cls(hidden_units=payload.get("hidden_units", (128, 64)))
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.feature_names = payload.get("feature_names", [])
        return instance
