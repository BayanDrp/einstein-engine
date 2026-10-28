
from dataclasses import dataclass

import numpy as np
from enum import Enum


class ObjectKind(str, Enum):
    PARTICLE = "particle"
    MATTER = "matter"

@dataclass
class Track:
    name: str
    times: np.ndarray
    positions: np.ndarray

    kind: ObjectKind = ObjectKind.PARTICLE
    display_size: float = 0.08
    color: str | None = None
    visible: bool = True

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float)

        if self.times.ndim != 1:
            raise ValueError("times must be a 1D array")

        if self.positions.shape != (len(self.times), 3):
            raise ValueError(
                "positions must have shape (N, 3)"
            )

        if len(self.times) == 0:
            raise ValueError("Track must contain at least one sample")

        if not np.isfinite(self.times).all():
            raise ValueError("times contains non finite values")

        if not np.isfinite(self.positions).all():
            raise ValueError("positions contains non finite values")

        if np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing")

        if not np.isfinite(self.display_size) or self.display_size <= 0:
            raise ValueError("display_size must be positive and finite")
