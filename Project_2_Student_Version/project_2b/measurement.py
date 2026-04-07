from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Measurement:
    """Unified measurement packet for simulator and real replay.

    Contract:
    - `old_contacts`: feet that were already in stance before this sample and are
      still in stance now (continuing contacts).
    - `new_contacts`: feet that transitioned swing -> stance at this sample.
    - Both dictionaries map foot id -> body-frame contact measurement `z_meas`.
    - The two dictionaries must be disjoint.
    - If both are empty, this sample is effectively IMU-only.
    """

    k: int
    dt: float
    omega_meas: np.ndarray
    f_meas: np.ndarray
    old_contacts: Dict[str, np.ndarray]
    new_contacts: Dict[str, np.ndarray]
