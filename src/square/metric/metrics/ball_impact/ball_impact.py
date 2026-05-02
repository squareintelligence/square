from typing import Dict, Protocol

import numpy as np

from square.metric.metric import Metric
from square.utils import load_joblib_model_from_s3


class ProbabilisticModel(Protocol):
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        ...

class BallImpact(Metric):
    def __init__(self, first_innings_model: str, second_innings_model: str):
        self.first_innings_model = load_joblib_model_from_s3(first_innings_model)
        self.second_innings_model = load_joblib_model_from_s3(second_innings_model)

    def compute(self, input_vars: Dict[str, np.ndarray]) -> np.ndarray:
        ball_runs = input_vars["ball_runs"]
        is_wicket = input_vars["is_wicket"]

        runs_before = input_vars["innings_runs"]
        runs_after = runs_before + ball_runs

        wickets_before = input_vars["wickets"]
        wickets_after = wickets_before + is_wicket

        balls_before = input_vars["balls"]
        is_legal = input_vars.get("is_legal")
        if is_legal is None:
            legal_inc = np.ones_like(balls_before, dtype=np.int64)
        else:
            legal_inc = np.asarray(is_legal != 0, dtype=np.int64)
        balls_after = balls_before + legal_inc

        target = input_vars["target"]
        runs_required_before = target - runs_before
        runs_required_after = target - runs_after

        second_innings_mask = target > 0
        first_innings_mask = ~second_innings_mask

        win_prob_before = np.zeros_like(runs_before, dtype=float)
        win_prob_after = np.zeros_like(runs_before, dtype=float)

        if np.any(first_innings_mask):
            first_before_features = np.column_stack(
                (
                    runs_before[first_innings_mask],
                    wickets_before[first_innings_mask],
                    balls_before[first_innings_mask],
                    runs_before[first_innings_mask] / np.maximum(balls_before[first_innings_mask], 1),
                )
            )
            first_after_features = np.column_stack(
                (
                    runs_after[first_innings_mask],
                    wickets_after[first_innings_mask],
                    balls_after[first_innings_mask],
                    runs_after[first_innings_mask] / np.maximum(balls_after[first_innings_mask], 1),
                )
            )
            win_prob_before[first_innings_mask] = self._predict_win_prob(
                self.first_innings_model, first_before_features
            )
            win_prob_after[first_innings_mask] = self._predict_win_prob(
                self.first_innings_model, first_after_features
            )

        if np.any(second_innings_mask):
            second_before_features = np.column_stack(
                (
                    target[second_innings_mask],
                    runs_required_before[second_innings_mask],
                    wickets_before[second_innings_mask],
                    balls_before[second_innings_mask],
                    runs_required_before[second_innings_mask]
                    / np.maximum(balls_before[second_innings_mask], 1),
                )
            )
            second_after_features = np.column_stack(
                (
                    target[second_innings_mask],
                    runs_required_after[second_innings_mask],
                    wickets_after[second_innings_mask],
                    balls_after[second_innings_mask],
                    runs_required_after[second_innings_mask]
                    / np.maximum(balls_after[second_innings_mask], 1),
                )
            )
            win_prob_before[second_innings_mask] = self._predict_win_prob(
                self.second_innings_model, second_before_features
            )
            win_prob_after[second_innings_mask] = self._predict_win_prob(
                self.second_innings_model, second_after_features
            )

        return win_prob_after - win_prob_before

    def _predict_win_prob(self, model: ProbabilisticModel, features: np.ndarray) -> np.ndarray:
        return model.predict_proba(features)[:, 1]