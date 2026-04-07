"""This script is responsible for the plotting logic of project 2a.
Do not modify the code in this file. Contact the course staff if you have any plotting issues."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gtsam


class LegPlotter:
    def __init__(
        self,
        T_END=5.0,
        L1=2.0,
        L2=2.0,
        FINE_RES=800,
        MEAS_NOISE=0.05,
        MIN_POINTS=3,
        x0=2.5,
        y0=3.4,
    ):
        self.T_END = T_END
        self.L1, self.L2 = L1, L2
        self.FINE_RES = FINE_RES
        self.MEAS_NOISE = MEAS_NOISE
        self.MIN_POINTS = MIN_POINTS
        self.x0, self.y0 = x0, y0

        # clicked points and processed angles
        self.theta1_clicks = []
        self.theta2_clicks = []
        self.theta1_lock_x = []
        self.theta2_lock_x = []
        self.theta1_point_colors = []
        self.theta2_point_colors = []
        self.theta_1 = None
        self.theta_2 = None
        self._dragging = None
        self._selected = None

        # pose timing/color convention (task frame)
        self.pose_times = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.pose_colors = ["red", "deeppink", "gold", "tab:green", "tab:blue"]
        self.pose_markers = ["o", "s", "^", "D", "P"]

        # Task targets: start with swing (0->2s), then stance (3->4s)
        self.x_muy = [0.500, 2.500, 4.500, 3.167, 1.833]
        self.y_muy = [0.000, 0.700, 0.000, 0.000, 0.000]

        # fine time for plotting
        self.t_fine = np.linspace(0, self.T_END, self.FINE_RES)

        # figure setup
        self.fig = plt.figure(figsize=(14, 7))
        gs = GridSpec(
            nrows=2,
            ncols=2,
            width_ratios=[1.2, 1.0],
            height_ratios=[1, 1],
            hspace=0.25,
            wspace=0.25,
        )
        self.fig.suptitle("2R Robot Leg Trajectory")
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax_fk = self.fig.add_subplot(gs[:, 1])
        self._setup_plot()
        self._initialize_seed_points()
        self.refresh()

        # mouse connection
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _setup_plot(self):
        """Setup plot elements."""
        # setup axes labels
        self.ax1.set_ylabel("theta_1 (deg)")
        self.ax1.set_xlabel("time (s)")
        self.ax2.set_ylabel("theta_2 (deg)")
        self.ax2.set_xlabel("time (s)")
        self.ax_fk.set_ylabel("y (m)")
        self.ax_fk.set_xlabel("x (m)")

        # setup axes limits
        self.ax1.set_xlim(0, self.T_END)
        self.ax2.set_xlim(0, self.T_END)
        self.ax1.set_ylim(-180, 180)
        self.ax2.set_ylim(-180, 180)
        self.ax_fk.set_xlim(-1.0, 5.0)
        self.ax_fk.set_ylim(-1.0, 5.0)
        self.ax_fk.set_aspect("equal", adjustable="box")

        # scatter and lines
        self.sc1 = self.ax1.scatter([], [], s=50)
        self.sc2 = self.ax2.scatter([], [], s=50)
        (self.line1,) = self.ax1.plot([], [], color="blue", lw=2)
        (self.line2,) = self.ax2.plot([], [], color="blue", lw=2)
        (self.fk_line,) = self.ax_fk.plot([], [], color="blue", lw=2)
        (self.link1,) = self.ax_fk.plot(
            [], [], lw=4, color="black", solid_capstyle="round", zorder=5
        )
        (self.link2,) = self.ax_fk.plot(
            [], [], lw=4, color="black", solid_capstyle="round", zorder=6
        )

        # quiver for FK arrows and time ticks
        self.quiver = self.ax_fk.quiver(
            [],
            [],
            [],
            [],
            angles="xy",
            scale_units="xy",
            scale=5,
            width=0.02,
            color="black",
        )
        self.ticks = None
        self.ghost_link_artists = []

        # fk plot elements
        self.ax_fk.axhline(0.0, linewidth=2, linestyle="--", label="Ground")
        self.ax_fk.set_aspect("equal", adjustable="box")
        self.ax_fk.set_anchor("C")
        self.ax_fk.set_xlabel("x (m)")
        self.ax_fk.set_ylabel("y (m)")
        self.start_fk_marker = self.ax_fk.scatter(
            [], [], s=120, color="red", zorder=5, label="Start/End"
        )
        self.base_marker = self.ax_fk.scatter(
            [self.x0], [self.y0], s=120, color="black", marker="x", zorder=6
        )
        for i, ((x_t, y_t), marker) in enumerate(
            zip(zip(self.x_muy, self.y_muy), self.pose_markers)
        ):
            color = self.pose_colors[i]
            self.ax_fk.scatter(
                [x_t],
                [y_t],
                s=80,
                color=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )
        self.ax_fk.axhline(
            0.0, linewidth=2, linestyle="--", color="black", label="Ground"
        )

        # turn on grid
        self.ax1.grid(True)
        self.ax2.grid(True)
        self.ax_fk.grid(True)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _get_pose_time_color_pairs(self):
        return [
            (t, c)
            for t, c in zip(self.pose_times, self.pose_colors)
            if 0.0 <= t <= self.T_END
        ]

    def _get_intermediate_tick_time_color_pairs(self):
        return [
            (t, c)
            for t, c in zip(self.pose_times[1:], self.pose_colors[1:])
            if 0.0 < t < self.T_END
        ]

    def _initialize_seed_points(self):
        """Seed both theta plots at t=0..4 with zero joint angles."""
        self.theta1_clicks = []
        self.theta2_clicks = []
        self.theta1_lock_x = []
        self.theta2_lock_x = []
        self.theta1_point_colors = []
        self.theta2_point_colors = []

        for t_pose, color in self._get_pose_time_color_pairs():
            self.theta1_clicks.append([t_pose, 0.0])
            self.theta2_clicks.append([t_pose, 0.0])
            self.theta1_lock_x.append(True)
            self.theta2_lock_x.append(True)
            self.theta1_point_colors.append(color)
            self.theta2_point_colors.append(color)

    def fit_fourier(self, points) -> np.ndarray:
        """Fit a GTSAM Fourier basis model to clicked points and return finer resolution values."""
        points = np.asarray(points, dtype=float)
        t = points[:, 0]
        y = points[:, 1]

        # Map t in [0, T_END] to x in [0, 2*pi]
        x = 2.0 * np.pi * (t / self.T_END)

        # GTSAM expects a map/dict of x -> y
        sequence = {float(xi): float(yi) for xi, yi in zip(x, y)}

        # Obtain the fitted parameters
        model = gtsam.noiseModel.Isotropic.Sigma(1, self.MEAS_NOISE)
        fit = gtsam.FitBasisFourierBasis(sequence, model, max(3, len(y)))  # type: ignore
        params = fit.parameters()

        x_fine = 2.0 * np.pi * (np.asarray(self.t_fine, dtype=float) / self.T_END)
        W = gtsam.FourierBasis.WeightMatrix(len(params), x_fine)
        y_fine = W @ params
        return np.asarray(y_fine).reshape(-1)

    def compute_fk(self, theta_1: np.ndarray, theta_2: np.ndarray):
        """Compute forward kinematics for 2R robot leg with 0
        angle pointing downward (-y)."""

        theta_1_shifted = theta_1 - np.pi / 2
        x1 = self.x0 + self.L1 * np.cos(theta_1_shifted)
        y1 = self.y0 + self.L1 * np.sin(theta_1_shifted)

        theta_2_shifted = theta_1_shifted + theta_2
        x2 = x1 + self.L2 * np.cos(theta_2_shifted)
        y2 = y1 + self.L2 * np.sin(theta_2_shifted)
        return x1, y1, x2, y2

    def _set_fk_arrows(self, x_fk, y_fk):
        dx = np.gradient(x_fk)
        dy = np.gradient(y_fk)
        norm = np.hypot(dx, dy)
        norm[norm == 0] = 1.0  # Avoid division by zero
        dx /= norm
        dy /= norm

        idx = np.arange(0, len(x_fk), 100)

        X = x_fk[idx]
        Y = y_fk[idx]
        U = dx[idx]
        V = dy[idx]

        # Reinitialize quiver safely
        self.quiver.remove()
        self.quiver = self.ax_fk.quiver(
            X,
            Y,
            U,
            V,
            angles="xy",
            scale_units="xy",
            scale=5,
            width=0.02,
            color="black",
            zorder=6,
        )

    def _set_fk_ticks(self, x_fk, y_fk):
        """Add time ticks to FK trajectory at 1s intervals."""
        tick_pairs = self._get_intermediate_tick_time_color_pairs()
        tick_indices = [np.argmin(np.abs(self.t_fine - t)) for t, _ in tick_pairs]
        tick_colors = [c for _, c in tick_pairs]

        if self.ticks is not None:
            self.ticks.remove()

        if len(tick_indices) == 0:
            self.ticks = None
            return

        self.ticks = self.ax_fk.scatter(
            x_fk[tick_indices], y_fk[tick_indices], color=tick_colors, s=70, zorder=5
        )

    def _get_intermediate_tick_times(self):
        """Intermediate FK times where markers and ghost legs are drawn."""
        return [t for t, _ in self._get_intermediate_tick_time_color_pairs()]

    def _get_intermediate_tick_colors(self):
        return [c for _, c in self._get_intermediate_tick_time_color_pairs()]

    def _get_intermediate_tick_indices(self):
        """Indices on self.t_fine corresponding to intermediate FK times."""
        tick_times = self._get_intermediate_tick_times()
        return [np.argmin(np.abs(self.t_fine - t)) for t in tick_times]

    def _draw_leg(
        self, x1, y1, x2, y2, lw=4, color="black", alpha=1.0, zorder=6, artists=None
    ):
        """Draw or update a single 2-link leg pose."""
        if artists is None:
            (link1,) = self.ax_fk.plot(
                [self.x0, x1],
                [self.y0, y1],
                lw=lw,
                color=color,
                alpha=alpha,
                solid_capstyle="round",
                zorder=zorder,
            )
            (link2,) = self.ax_fk.plot(
                [x1, x2],
                [y1, y2],
                lw=lw,
                color=color,
                alpha=alpha,
                solid_capstyle="round",
                zorder=zorder,
            )
            return link1, link2

        link1, link2 = artists
        link1.set_data([self.x0, x1], [self.y0, y1])
        link2.set_data([x1, x2], [y1, y2])
        link1.set_alpha(alpha)
        link2.set_alpha(alpha)
        return link1, link2

    def _set_ghost_legs(self, x1, y1, x2, y2):
        """Draw faint intermediate leg poses at fixed tick times."""
        tick_indices = self._get_intermediate_tick_indices()
        tick_colors = self._get_intermediate_tick_colors()

        for artist in self.ghost_link_artists:
            artist.remove()
        self.ghost_link_artists = []

        for idx, color in zip(tick_indices, tick_colors):
            ghost_link1, ghost_link2 = self._draw_leg(
                x1[idx],
                y1[idx],
                x2[idx],
                y2[idx],
                lw=3,
                color=color,
                alpha=0.75,
                zorder=4,
            )
            self.ghost_link_artists.extend((ghost_link1, ghost_link2))

    def refresh(self):
        """Refresh the plot with current clicked points and fitted curves."""
        # Update scatter plots
        if self.theta1_clicks:
            pts = np.array(self.theta1_clicks)
            pts_deg = pts.copy()
            pts_deg[:, 1] = np.degrees(pts_deg[:, 1])
            self.sc1.set_offsets(pts_deg)
            self.sc1.set_color(self.theta1_point_colors)
        else:
            self.sc1.set_offsets(np.empty((0, 2)))
        if self.theta2_clicks:
            pts = np.array(self.theta2_clicks)
            pts_deg = pts.copy()
            pts_deg[:, 1] = np.degrees(pts_deg[:, 1])
            self.sc2.set_offsets(pts_deg)
            self.sc2.set_color(self.theta2_point_colors)
        else:
            self.sc2.set_offsets(np.empty((0, 2)))

        # update theta_1 if we have enough points
        if len(self.theta1_clicks) >= self.MIN_POINTS:
            th1 = self.fit_fourier(self.theta1_clicks)
            self.theta_1 = np.array([a for a in th1])
            self.line1.set_data(self.t_fine, np.degrees(self.theta_1))

        # update theta_2 if we have enough points
        if len(self.theta2_clicks) >= self.MIN_POINTS:
            th2 = self.fit_fourier(self.theta2_clicks)
            self.theta_2 = np.array([a for a in th2])
            self.line2.set_data(self.t_fine, np.degrees(self.theta_2))

        # update FK plot if both theta_1 and theta_2 are fitted
        if (self.theta_1 is not None) and (self.theta_2 is not None):
            x1, y1, x2, y2 = self.compute_fk(self.theta_1, self.theta_2)
            self.fk_line.set_data(x2, y2)
            self._set_fk_arrows(x2, y2)
            self._set_fk_ticks(x2, y2)
            self._set_ghost_legs(x1, y1, x2, y2)

            # start/end marker and legs
            self.start_fk_marker.set_offsets([[x2[0], y2[0]]])
            self._draw_leg(
                x1[-1],
                y1[-1],
                x2[-1],
                y2[-1],
                artists=(self.link1, self.link2),
            )

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Handle mouse click events to add points."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        t_click = float(event.xdata)
        y_click = np.radians(float(event.ydata))
        y_click = self.normalize_angle(y_click)

        threshold = 0.2

        # check if clicked point is near existing point for dragging
        if event.inaxes == self.ax1:
            pts = np.array(self.theta1_clicks)
            if len(pts) > 0:
                dists = np.hypot(pts[:, 0] - t_click, pts[:, 1] - y_click)
                idx = np.argmin(dists)
                if dists[idx] < threshold:
                    self._dragging = ("theta1", idx)
                    self._selected = ("theta1", idx)
                    return
            self.theta1_clicks.append([t_click, y_click])
            self.theta1_lock_x.append(False)
            self.theta1_point_colors.append("black")
            self._selected = None
        elif event.inaxes == self.ax2:
            pts = np.array(self.theta2_clicks)
            if len(pts) > 0:
                dists = np.hypot(pts[:, 0] - t_click, pts[:, 1] - y_click)
                idx = np.argmin(dists)
                if dists[idx] < threshold:
                    self._dragging = ("theta2", idx)
                    self._selected = ("theta2", idx)
                    return
            self.theta2_clicks.append([t_click, y_click])
            self.theta2_lock_x.append(False)
            self.theta2_point_colors.append("black")
            self._selected = None

        self.refresh()

    def on_drag(self, event):
        """Move the currently dragged point while mouse is moving."""
        if (
            self._dragging is None
            or event.inaxes is None
            or event.xdata is None
            or event.ydata is None
        ):
            return

        t_new = float(event.xdata)
        y_new = np.radians(float(event.ydata))
        y_new = self.normalize_angle(y_new)

        kind, idx = self._dragging
        if kind == "theta1":
            if self.theta1_lock_x[idx]:
                t_new = self.theta1_clicks[idx][0]
            self.theta1_clicks[idx] = [t_new, y_new]
        elif kind == "theta2":
            if self.theta2_lock_x[idx]:
                t_new = self.theta2_clicks[idx][0]
            self.theta2_clicks[idx] = [t_new, y_new]

        self.refresh()

    def on_release(self, event):
        """Stop dragging when mouse button is released."""
        self._dragging = None

    def on_key(self, event):
        """Handle key press events (backspace deletes selected point)."""
        if (event.key != "backspace") or self._selected is None:
            return

        kind, idx = self._selected

        if kind == "theta1" and 0 <= idx < len(self.theta1_clicks):
            del self.theta1_clicks[idx]
            del self.theta1_lock_x[idx]
            del self.theta1_point_colors[idx]

        elif kind == "theta2" and 0 <= idx < len(self.theta2_clicks):
            del self.theta2_clicks[idx]
            del self.theta2_lock_x[idx]
            del self.theta2_point_colors[idx]

        self._selected = None
        self._dragging = None
        self.refresh()


def launch():
    """Launch interactive robot plot with arrows in FK."""
    plotter = LegPlotter()
    plt.show()
    return plotter
