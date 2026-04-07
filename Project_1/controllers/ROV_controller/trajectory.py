from dataclasses import dataclass
import math

import gtsam
from gtsam import NavState
import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Circular trajectory with vertical oscillation.

    Parameters:
        T: Period for one full horizontal revolution in seconds.
        r: Horizontal radius in meters.
        z_amp: Vertical oscillation amplitude in meters.
        z_freq_mult: Multiplier on the base angular frequency for z oscillation.
        center: (x, y, z) center of the circle in meters.
    """

    T: float = 80.0
    r: float = 2.0
    z_amp: float = 0.2
    z_freq_mult: float = 5.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.6)

    def query(self, t: float) -> NavState:
        omega = 2.0 * math.pi / self.T
        xc, yc, zc = self.center
        x = xc + self.r * math.cos(omega * t)
        y = yc + self.r * math.sin(omega * t)
        z = zc + self.z_amp * math.cos(self.z_freq_mult * omega * t)
        pos = np.array([x, y, z], dtype=float)
        vx = -self.r * omega * math.sin(omega * t)
        vy = self.r * omega * math.cos(omega * t)
        vz = -self.z_amp * self.z_freq_mult * omega * math.sin(
            self.z_freq_mult * omega * t
        )
        yaw = math.atan2(vy, vx)
        X_des = gtsam.NavState(
            gtsam.Rot3.Yaw(yaw),
            gtsam.Point3(pos),
            np.array([vx, vy, vz], dtype=float),
        )
        return X_des
