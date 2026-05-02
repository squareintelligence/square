from square.metric.metric import Metric
import numpy as np
from typing import Dict

class BallImpact(Metric):
    def __init__(self, model: str):
        self.model_name = model

    def compute(self, input_vars: Dict[str, np.ndarray]) -> np.ndarray:
        ball_runs = input_vars["ball_runs"]
        is_wicket = input_vars["is_wicket"]

        runs_before = input_vars["innings_runs"]
        runs_after = runs_before + ball_runs

        wickets_before = input_vars["wickets"]
        wickets_after = wickets_before + is_wicket

        balls = input_vars["balls"]
        balls_after = balls + 1

        target = input_vars["target"]
        runs_required_before = target - runs_before
        runs_required_per_ball_before = runs_required_before / balls

        runs_required_after = target - runs_after
        runs_required_per_ball_after = runs_required_after / balls_after

        return 0