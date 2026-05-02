from typing import Protocol, Dict
from square.metric.registry import MetricRegistry

import numpy as np


class Metric(Protocol):
    def compute(self, input_vars: Dict[str, np.ndarray]) -> np.ndarray:
        ...

def build_metric(name: str, params: dict) -> Metric:
    if name not in MetricRegistry:
        raise ValueError(f"Unknown metric: {name}")

    return MetricRegistry[name](**params)