from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from rosbags.highlevel import AnyReader

try:
    from .measurement import Measurement
except ImportError:
    from measurement import Measurement


class RealReplay:
    """Stream IMU/contact messages from a bag and emit Measurement objects online."""

    def __init__(self, bag_path: str, contact_state_value: int | None = None):
        self.bag_path = Path(bag_path)
        self.foot_names = ["FL", "FR", "RL", "RR"]
        self.contact_state_value = contact_state_value

        self._reader = AnyReader([self.bag_path])  # type: ignore[arg-type]
        self._reader.open()
        self._closed = False

        self._imu_connections = [c for c in self._reader.connections if c.topic == "/imu"]
        self._foot_connections = [
            c for c in self._reader.connections if c.topic == "/spot/status/feet"
        ]
        self._replay_connections = self._imu_connections + self._foot_connections

        if not self._imu_connections:
            raise ValueError("No /imu topic found in bag.")
        if not self._foot_connections:
            raise ValueError("No /spot/status/feet topic found in bag.")

        # Running state statistics for auto-detecting which contact-state code is stance.
        self._state_counts: dict[int, int] = {}
        self._state_mean_z: dict[int, float] = {}
        self._min_state_observations = 8

        self._reset_stream_state()

    def __enter__(self) -> RealReplay:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._reader.close()
        except Exception:
            pass
        self._closed = True

    @staticmethod
    def _msg_time_s(msg) -> float:
        return float(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)  # type: ignore[attr-defined]

    def _reset_stream_state(self) -> None:
        self._k = 0
        self._last_sample_time_s: float | None = None
        self._last_imu_time_s: float | None = None
        self._last_contact_emit_time_s: float | None = None
        self._last_omega = np.zeros(3, dtype=float)
        self._last_f = np.zeros(3, dtype=float)
        self._has_imu = False

        self._prev_raw_contacts: dict[str, int | None] = {
            foot: None for foot in self.foot_names
        }

    def _next_dt(self, current_time_s: float) -> float:
        if self._last_sample_time_s is None:
            return 0.01
        return max(current_time_s - self._last_sample_time_s, 0.0)

    def _emit_sample(
        self,
        timestamp_s: float,
        old_contacts: dict[str, np.ndarray],
        new_contacts: dict[str, np.ndarray],
        callback: Callable[[Measurement], None],
    ) -> None:
        measurement = Measurement(
            k=self._k,
            dt=self._next_dt(timestamp_s),
            omega_meas=self._last_omega.copy(),
            f_meas=self._last_f.copy(),
            old_contacts=old_contacts,
            new_contacts=new_contacts,
        )
        callback(measurement)
        self._k += 1
        self._last_sample_time_s = timestamp_s

    def _update_contact_state_stats(self, state_value: int, z_body: float) -> None:
        count = self._state_counts.get(state_value, 0) + 1
        prev_mean = self._state_mean_z.get(state_value, 0.0)
        mean = prev_mean + (z_body - prev_mean) / count
        self._state_counts[state_value] = count
        self._state_mean_z[state_value] = mean

    def _maybe_infer_contact_state(self) -> None:
        if self.contact_state_value is not None:
            return

        candidates = [
            (self._state_mean_z[state], state)
            for state, count in self._state_counts.items()
            if count >= self._min_state_observations
        ]
        if len(candidates) < 2:
            return

        # In body coordinates, stance feet tend to have lower (more negative) z.
        self.contact_state_value = min(candidates)[1]

    def _process_imu_message(self, msg, callback: Callable[[Measurement], None]) -> None:
        t_imu = self._msg_time_s(msg)

        self._last_imu_time_s = t_imu
        self._last_omega = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],  # type: ignore[attr-defined]
            dtype=float,
        )
        self._last_f = np.array(
            [
                msg.linear_acceleration.x,  # type: ignore[attr-defined]
                msg.linear_acceleration.y,  # type: ignore[attr-defined]
                msg.linear_acceleration.z,  # type: ignore[attr-defined]
            ],
            dtype=float,
        )
        self._has_imu = True

        self._emit_sample(
            timestamp_s=t_imu,
            old_contacts={},
            new_contacts={},
            callback=callback,
        )

    def _process_foothold_message(
        self, msg, callback: Callable[[Measurement], None]
    ) -> None:
        t_contact = self._msg_time_s(msg)
        states = list(msg.states)  # type: ignore[attr-defined]

        raw_contacts: dict[str, int] = {}
        foot_measurements: dict[str, np.ndarray] = {}

        for i, foot_state in enumerate(states):
            if i >= len(self.foot_names):
                break
            foot = self.foot_names[i]
            raw = int(foot_state.contact)
            x = float(foot_state.foot_position_rt_body.x)
            y = float(foot_state.foot_position_rt_body.y)
            z = float(foot_state.foot_position_rt_body.z)

            raw_contacts[foot] = raw
            foot_measurements[foot] = np.array([x, y, z], dtype=float)
            self._update_contact_state_stats(raw, z)

        self._maybe_infer_contact_state()

        start_feet: set[str] = set()
        continuing_feet: set[str] = set()
        contact_state = self.contact_state_value

        if contact_state is not None:
            for foot, curr_raw in raw_contacts.items():
                prev_raw = self._prev_raw_contacts[foot]
                if prev_raw is not None:
                    prev_contact = prev_raw == contact_state
                    curr_contact = curr_raw == contact_state
                    if curr_contact and not prev_contact:
                        start_feet.add(foot)
                    if curr_contact and prev_contact:
                        continuing_feet.add(foot)

        for foot, curr_raw in raw_contacts.items():
            self._prev_raw_contacts[foot] = curr_raw

        # Emit foothold updates only on swing->stance transitions.
        if not start_feet or not self._has_imu:
            return
        if (
            self._last_contact_emit_time_s is not None
            and abs(t_contact - self._last_contact_emit_time_s) < 1e-12
        ):
            return

        old_contacts = {
            foot: foot_measurements[foot].copy()
            for foot in self.foot_names
            if foot in continuing_feet
        }
        new_contacts = {
            foot: foot_measurements[foot].copy()
            for foot in self.foot_names
            if foot in start_feet
        }

        self._emit_sample(
            timestamp_s=t_contact,
            old_contacts=old_contacts,
            new_contacts=new_contacts,
            callback=callback,
        )
        self._last_contact_emit_time_s = t_contact

    def replay(self, callback: Callable[[Measurement], None]) -> None:
        """Deliver interleaved IMU/contact measurements to a callback."""
        if self._closed:
            raise RuntimeError("RealReplay reader is closed.")

        self._reset_stream_state()
        for connection, _, rawdata in self._reader.messages(
            connections=self._replay_connections
        ):
            msg = self._reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == "/imu":
                self._process_imu_message(msg, callback)
            elif connection.topic == "/spot/status/feet":
                self._process_foothold_message(msg, callback)
