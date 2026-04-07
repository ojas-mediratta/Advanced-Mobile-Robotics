"""Logging and plotting helpers for ROV controller."""

from __future__ import annotations

import os
import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore


def _color_map():
    return {
        0: ("rgba(255,0,0,0.15)", "#ff0000"),
        1: ("rgba(0,128,0,0.15)", "#008000"),
        2: ("rgba(0,0,255,0.15)", "#0000ff"),
    }


class Logger:
    def __init__(self):
        self.times = []
        self.X_true = []
        self.X_est = []
        self.X_des = []
        self.P_std_list = []
        self.save_plots = False   # If True will save plots to current directory (enable for WSL2)
        self.script_dir = os.path.dirname(__file__)

    @staticmethod
    def attitudes(nav_states):
        """Return Nx3 yaw/pitch/roll (rad) for a list of NavStates."""
        if not nav_states:
            return np.empty((0, 3))
        return np.array([x.attitude().ypr() for x in nav_states])

    @staticmethod
    def ypr_deg(nav_states):
        """Return Nx3 yaw/pitch/roll (deg) for a list of NavStates."""
        if not nav_states:
            return np.empty((0, 3))
        return np.degrees(Logger.attitudes(nav_states))

    @staticmethod
    def positions(nav_states):
        """Return Nx3 positions for a list of NavStates."""
        if not nav_states:
            return np.empty((0, 3))
        return np.array([x.position() for x in nav_states], dtype=float)

    @staticmethod
    def velocities(nav_states):
        """Return Nx3 velocities for a list of NavStates."""
        if not nav_states:
            return np.empty((0, 3))
        return np.array([x.velocity() for x in nav_states], dtype=float)

    def log_state(self, t, X_true, X_est, P, desired_state=None):
        self.times.append(t)
        self.X_true.append(X_true)
        self.X_est.append(X_est)
        self.X_des.append(desired_state)

        self.P_std_list.append(np.sqrt(np.diag(P)[:9]).copy())

    def plot_attitude(self):
        if not self.X_true or not self.X_est or not self.P_std_list:
            return

        ypr_true_plt = np.unwrap(self.attitudes(self.X_true), axis=0)
        ypr_est_plt = np.unwrap(self.attitudes(self.X_est), axis=0)
        std = np.vstack(self.P_std_list)
        rot_std_deg_plt = np.rad2deg(std[:, 0:3])

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Yaw (deg)", "Pitch (deg)", "Roll (deg)"),
        )
        color_rgb = _color_map()
        names = {0: "Yaw", 1: "Pitch", 2: "Roll"}
        desired_yaw = None
        if any(x is not None for x in self.X_des):
            desired_yaw = np.full((len(self.X_des),), np.nan, dtype=float)
            for i, x_des in enumerate(self.X_des):
                if x_des is not None:
                    desired_yaw[i] = np.degrees(x_des.attitude().yaw())
        for i in range(3):
            fill_rgba, line_color = color_rgb[i]
            mean = np.degrees(ypr_est_plt[:, i])
            std2p = 2.0 * rot_std_deg_plt[:, i]
            upper = mean + std2p
            lower = mean - std2p
            row = i + 1
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=upper,
                    line=dict(color=line_color, width=0),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=lower,
                    fill="tonexty",
                    fillcolor=fill_rgba,
                    line=dict(color=line_color, width=0),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=mean,
                    name=f"{names[i]} est",
                    line=dict(color=line_color, width=2),
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=np.degrees(ypr_true_plt[:, i]),
                    name=f"{names[i]} true",
                    line=dict(color=line_color, dash="dash", width=2),
                ),
                row=row,
                col=1,
            )
            if i == 0 and desired_yaw is not None:
                fig.add_trace(
                    go.Scatter(
                        x=self.times,
                        y=desired_yaw,
                        name="Yaw des",
                        line=dict(color="#ff7f0e", dash="dot", width=2),
                    ),
                    row=row,
                    col=1,
                )
        fig.update_layout(
            height=900,
            title_text="Yaw, Pitch, Roll (deg) with +/-2sigma bounds",
            xaxis3_title="Time (s)",
            yaxis_title="Degrees",
            showlegend=True,
        )
        if self.save_plots: 
            fig.write_image(os.path.join(self.script_dir, 'attitude_plot.png'))
        else: 
            fig.show()

    def plot_position(self):
        if not self.X_true or not self.X_est or not self.P_std_list:
            return

        pos_true_plt = self.positions(self.X_true)
        pos_est_plt = self.positions(self.X_est)
        pos_std_plt = np.vstack(self.P_std_list)[:, 3:6]

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Position X (m)", "Position Y (m)", "Position Z (m)"),
        )
        color_rgb = _color_map()
        desired_pos = None
        if any(x is not None for x in self.X_des):
            desired_pos = np.full((len(self.times), 3), np.nan, dtype=float)
            for i, x_des in enumerate(self.X_des):
                if x_des is not None:
                    desired_pos[i, :] = np.asarray(x_des.position(), dtype=float)
        for i, comp in enumerate(["x", "y", "z"]):
            fill_rgba, line_color = color_rgb[i]
            mean = pos_est_plt[:, i]
            std2p = 2.0 * pos_std_plt[:, i]
            upper = mean + std2p
            lower = mean - std2p
            row = i + 1
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=upper,
                    line=dict(color=line_color, width=0),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=lower,
                    fill="tonexty",
                    fillcolor=fill_rgba,
                    line=dict(color=line_color, width=0),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=mean,
                    name=f"p{comp} est",
                    line=dict(color=line_color, width=2),
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=pos_true_plt[:, i],
                    name=f"p{comp} true",
                    line=dict(color=line_color, dash="dash", width=2),
                ),
                row=row,
                col=1,
            )
            if desired_pos is not None:
                fig.add_trace(
                    go.Scatter(
                        x=self.times,
                        y=desired_pos[:, i],
                        name=f"p{comp} des",
                        line=dict(color="#ff7f0e", dash="dot", width=2),
                    ),
                    row=row,
                    col=1,
                )
        fig.update_layout(
            height=900,
            title_text="Position Components with +/-2sigma Bounds",
            xaxis3_title="Time (s)",
            yaxis_title="Position (m)",
            showlegend=True,
        )
        if self.save_plots: 
            fig.write_image(os.path.join(self.script_dir, 'position_plot.png'))
        else: 
            fig.show()

    def plot_velocity(self):
        if not self.X_true or not self.X_est or not self.P_std_list:
            return

        vel_true_plt = self.velocities(self.X_true)
        vel_est_plt = self.velocities(self.X_est)
        vel_std_plt = np.vstack(self.P_std_list)[:, 6:9]

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Velocity X (m/s)", "Velocity Y (m/s)", "Velocity Z (m/s)"),
        )
        color_rgb = _color_map()
        for i, comp in enumerate(["x", "y", "z"]):
            fill_rgba, line_color = color_rgb[i]
            mean = vel_est_plt[:, i]
            std2p = 2.0 * vel_std_plt[:, i]
            upper = mean + std2p
            lower = mean - std2p
            row = i + 1
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=upper,
                    line=dict(color=line_color, width=0),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=lower,
                    fill="tonexty",
                    fillcolor=fill_rgba,
                    line=dict(color=line_color, width=0),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=mean,
                    name=f"v{comp} est",
                    line=dict(color=line_color, width=2),
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.times,
                    y=vel_true_plt[:, i],
                    name=f"v{comp} true",
                    line=dict(color=line_color, dash="dash", width=2),
                ),
                row=row,
                col=1,
            )
        fig.update_layout(
            height=900,
            title_text="Velocity Components in World Frame with +/-2sigma Bounds",
            xaxis3_title="Time (s)",
            yaxis_title="Velocity (m/s)",
            showlegend=True,
        )
        if self.save_plots: 
            fig.write_image(os.path.join(self.script_dir, 'velocity_plot.png'))
        else: 
            fig.show()

    def plot_trajectory_tracking(
        self,
        title: str = "Desired vs Actual Position",
    ):
        if not self.times or not self.X_true or not self.X_des:
            return
        if not any(x is not None for x in self.X_des):
            return

        t = np.asarray(self.times)
        pos = self.positions(self.X_true)
        pos_des = np.full((len(self.X_des), 3), np.nan, dtype=float)
        for i, x_des in enumerate(self.X_des):
            if x_des is not None:
                pos_des[i, :] = np.asarray(x_des.position(), dtype=float)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("X Position", "Y Position", "Z Position"),
        )

        labels = ["x", "y", "z"]

        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=pos_des[:, i],
                    mode="lines",
                    name=f"{labels[i]}_des",
                    line=dict(dash="dash"),
                ),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=pos[:, i],
                    mode="lines",
                    name=f"{labels[i]}",
                ),
                row=i + 1,
                col=1,
            )

        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Position [m]",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            title=title,
            margin=dict(l=60, r=30, t=60, b=50),
        )

        if self.save_plots: 
            fig.write_image(os.path.join(self.script_dir, 'trajectory_plot.png'))
        else: 
            fig.show()

    def plot_trajectory_tracking_3d(
        self,
        title: str = "3D Trajectory: Desired vs Actual",
        show_markers: bool = False,
    ):
        if not self.X_true or not self.X_des:
            return
        if not any(x is not None for x in self.X_des):
            return

        pos = self.positions(self.X_true)
        pos_des = np.full((len(self.X_des), 3), np.nan, dtype=float)
        for i, x_des in enumerate(self.X_des):
            if x_des is not None:
                pos_des[i, :] = np.asarray(x_des.position(), dtype=float)

        mode = "lines+markers" if show_markers else "lines"

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=pos_des[:, 0],
                y=pos_des[:, 1],
                z=pos_des[:, 2],
                mode=mode,
                name="Desired",
                line=dict(dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode=mode,
                name="Actual",
            )
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                zaxis_title="Z [m]",
                aspectmode="data",
            ),
            template="plotly_white",
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        if self.save_plots: 
            fig.write_image(os.path.join(self.script_dir, 'trajectory_3d_plot.png'))
        else: 
            fig.show()
