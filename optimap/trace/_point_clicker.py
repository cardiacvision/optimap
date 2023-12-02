# Modified version of mpl-point-clicker: https://github.com/mpl-extensions/mpl-point-clicker/

# Original license:
# Copyright (c) Ian Hunt-Isaak.
# Distributed under the terms of the Modified BSD License.

import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.cbook import CallbackRegistry
from matplotlib.pyplot import cm


class PointClicker:
    def __init__(self, ax, as_integer=False, single_point=False):
        """Parameters
        ----------
        ax : matplotlib axis
        integer : bool, return integer positions if True
        single_point : bool, only allow one point if True
        """
        self.ax = ax
        self._integer = as_integer
        self._single_point = single_point
        self._scat = self.ax.scatter([], [])
        self._fig = self.ax.figure
        self._fig.canvas.mpl_connect("button_press_event", self._clicked)
        self._positions = []
        self._observers = CallbackRegistry()

    def get_positions(self):
        return self._positions

    def set_positions(self, positions):
        """Set the current positions.

        Parameters
        ----------
        positions : list of tuples
            List of positions to set.
        """
        self._positions = list(positions)
        self._observers.process("pos-set", self.get_positions())

    def _clicked(self, event):
        if not self._fig.canvas.widgetlock.available(self):
            return
        if event.inaxes is self.ax:
            ix, iy = event.xdata, event.ydata
            if self._integer:
                ix, iy = int(round(ix)), int(round(iy))
            if event.button is MouseButton.LEFT:
                if (ix, iy) not in self._positions:
                    if self._single_point:
                        self._positions = []
                    self._positions.append((ix, iy))
                    self._update_points()
                    self._observers.process(
                        "point-added",
                        (ix, iy),
                    )
                else:
                    idx = self._positions.index((ix, iy))
                    removed = self._positions.pop(idx)
                    self._update_points()
                    self._observers.process(
                        "point-removed",
                        removed,
                        idx,
                    )
            elif event.button is MouseButton.RIGHT:
                pos = self._positions
                if len(pos) == 0:
                    return
                dists = np.linalg.norm(
                    np.asarray([ix, iy])[None, None, :]
                    - np.asarray(pos)[None, :, :],
                    axis=-1,
                )
                idx = np.argmin(dists[0])
                removed = pos.pop(idx)
                self._update_points()
                self._observers.process(
                    "point-removed",
                    removed,
                    idx,
                )

    def _update_points(self):
        pos = np.array(self._positions)
        colors = cm.tab10.colors
        while len(colors) < len(pos):
            colors += colors
        colors = colors[:len(pos)]

        try:
            self._scat.remove()
        except ValueError:  # raises an error when scatter empty
            pass

        if len(pos) == 0:
            self._scat = self.ax.scatter([], [])
        else:
            self._scat = self.ax.scatter(pos[:, 0], pos[:, 1], c=colors)
        self._fig.canvas.draw()

    def on_point_added(self, func):
        """Connect *func* as a callback function to new points being added.

        *func* will receive the the position of the new point as a tuple (x, y).

        Parameters
        ----------
        func : callable
            Function to call when a point is added.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect("point-added", lambda *args: func(*args))

    def on_point_removed(self, func):
        """Connect *func* as a callback function when points are removed.

        *func* will receive the the position of the new point, the point's index in the old list of points, and the
        updated dictionary of all points.

        Parameters
        ----------
        func : callable
            Function to call when a point is removed

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect("point-removed", lambda *args: func(*args))

    def on_positions_set(self, func):
        """Connect *func* as a callback function when the *set_positions* function is
        called.

        *func* will receive the updated dictionary of all points.

        Parameters
        ----------
        func : callable
            Function to call when *set_positions* is called.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect("pos-set", lambda pos_dict: func(pos_dict))
