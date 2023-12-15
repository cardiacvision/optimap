import pathlib
from collections import deque

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, LassoSelector
from mpl_pan_zoom import PanManager, zoom_factory
from PIL import Image

ASSETS_DIR = pathlib.Path(__file__).parent.parent / "assets"
DRAW_SYMBOL = ASSETS_DIR / "draw_symbol.png"
ERASER_SYMBOL = ASSETS_DIR / "eraser_symbol.png"
UNDO_SYMBOL = ASSETS_DIR / "undo_symbol.png"
REDO_SYMBOL = ASSETS_DIR / "redo_symbol.png"
VISIBILITY_ON_SYMBOL = ASSETS_DIR / "visibility_on_symbol.png"
VISIBILITY_OFF_SYMBOL = ASSETS_DIR / "visibility_off_symbol.png"
INVERT_SYMBOL = ASSETS_DIR / "invert_symbol.png"

class ImageSegmenter:
    """Manually segment an image with the lasso selector."""

    _mask_history = deque()
    _mask_future = deque()

    def __init__(  # type: ignore
        self,
        image,
        mask=None,
        default_tool="draw",
        mask_color="red",
        mask_alpha=0.3,
        lasso_props=None,
        lasso_mousebutton="left",
        pan_mousebutton="middle",
        ax=None,
        title="",
        figsize=(7, 7),
        **kwargs,
    ):
        """Manually segment an image.

        .. table:: **Keyboard Shortcuts**

            ========================= ===========================
            Key                       Action
            ========================= ===========================
            ``Scroll``                Zoom in/out
            ``ctrl+z`` or ``cmd+z``   Undo
            ``ctrl+y`` or ``cmd+y``   Redo
            ``e``                     Erase mode
            ``d``                     Draw/Lasso mode
            ``v``                     Toggle mask visibility
            ``q``                     Quit
            ========================= ===========================

        Parameters
        ----------
        image : array_like
            A single image to segment
        mask : arraylike, optional
            If you want to pre-seed the mask
        default_tool : str, default: "draw"
            The default tool to use. One of "draw" or "erase"
        mask_color : None, color, or array of colors, optional
            the colors to use for each class. Unselected regions will always be
            totally transparent
        mask_alpha : float, default .3
            The alpha values to use for selected regions. This will always override
            the alpha values in mask_colors if any were passed
        lasso_props : dict, default: None
            props passed to LassoSelector. If None the default values are:
            {"color": mask_color, "linewidth": 1, "alpha": 0.8}
        lasso_mousebutton : str, or int, default: "left"
            The mouse button to use for drawing the selecting lasso.
        pan_mousebutton : str, or int, default: "middle"
            The button to use for `~mpl_interactions.generic.panhandler`. One of
            'left', 'middle', 'right', or 1, 2, 3 respectively.
        ax : `matplotlib.axes.Axes`, optional
            The axis on which to plot. If *None* a new figure will be created.
        title : str, optional
            The title of the plot.
        figsize : (float, float), optional
            passed to plt.figure. Ignored if *ax* is given.
        **kwargs : dict
            All other kwargs will passed to the imshow command for the image
        """
        if image.ndim != 2 and image.shape[-1] != 3:
            msg = "Image must be an image, got shape {image.shape}"
            raise ValueError(msg)

        self._img = image
        if default_tool == "draw":
            self._erasing = False
        elif default_tool == "erase":
            self._erasing = True
        else:
            raise ValueError(f"Unknown default tool '{default_tool}'. Must be 'draw' or 'erase'")
        self._visible = True
        self.cmap_mask = mpl.colors.ListedColormap(["none", mask_color])

        if mask is None:
            self.mask = np.zeros(self._img.shape[:2], dtype=bool)
        else:
            self.mask = mask.astype(bool)

        if ax is not None:
            self.ax = ax
            self.fig = self.ax.figure
        else:
            with plt.ioff():
                self.fig, self.ax = plt.subplots(figsize=figsize)
        self._displayed = self.ax.imshow(self._img, interpolation="none", **kwargs)
        self._mask_im = self.ax.imshow(
            self._mask,
            cmap=self.cmap_mask,
            vmin=0,
            vmax=1,
            alpha=mask_alpha,
            interpolation="none",
        )
        if title:
            self.ax.set_title(title)

        default_lasso_props = {"color": mask_color, "linewidth": 1, "alpha": 0.8}
        if lasso_props is None:
            lasso_props = default_lasso_props

        useblit = False if "ipympl" in mpl.get_backend().lower() else True
        button_dict = {"left": 1, "middle": 2, "right": 3}
        if isinstance(pan_mousebutton, str):
            pan_mousebutton = button_dict[pan_mousebutton.lower()]
        if isinstance(lasso_mousebutton, str):
            lasso_mousebutton = button_dict[lasso_mousebutton.lower()]

        if mpl.__version__ < "3.7":
            self.lasso = LassoSelector(
                self.ax,
                self._onselect,
                lineprops=lasso_props,
                useblit=useblit,
                button=lasso_mousebutton,
            )
        else:
            self.lasso = LassoSelector(
                self.ax,
                self._onselect,
                props=lasso_props,
                useblit=useblit,
                button=lasso_mousebutton,
            )
        self.lasso.set_visible(True)
        pix_x = np.arange(self._img.shape[0])
        pix_y = np.arange(self._img.shape[1])
        xv, yv = np.meshgrid(pix_y, pix_x)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T

        self._pm = PanManager(self.fig, button=pan_mousebutton)
        self.disconnect_zoom = zoom_factory(self.ax)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        self._setup_gui()

    def _setup_gui(self) -> None:
        BUTTON_WIDTH = 0.05
        BUTTON_SPACING = 0.025
        pos = [0.22, 0.92, BUTTON_WIDTH, BUTTON_WIDTH]

        ax_undo = self.fig.add_axes(pos)
        self.button_undo = Button(ax_undo, "", image=Image.open(UNDO_SYMBOL))
        self.button_undo.on_clicked(self._undo)

        pos[0] += BUTTON_WIDTH + BUTTON_SPACING
        ax_redo = self.fig.add_axes(pos)
        self.button_redo = Button(ax_redo, "", image=Image.open(REDO_SYMBOL))
        self.button_redo.on_clicked(self._redo)

        pos[0] += BUTTON_WIDTH + BUTTON_SPACING
        self._ax_visibility = self.fig.add_axes(pos)
        self.button_viz = Button(
            self._ax_visibility, "", image=Image.open(VISIBILITY_ON_SYMBOL)
        )
        self.button_viz.on_clicked(self._toggle_visibility)

        pos[0] += BUTTON_WIDTH + BUTTON_SPACING
        ax_inverse = self.fig.add_axes(pos)
        self.button_invert = Button(
            ax_inverse, "", image=Image.open(INVERT_SYMBOL)
        )
        self.button_invert.on_clicked(self._inverse_mask)

        pos[0] += BUTTON_WIDTH + BUTTON_SPACING
        ax_draw = self.fig.add_axes(pos)
        self.button_draw = Button(ax_draw, "", image=Image.open(DRAW_SYMBOL))
        self.button_draw.on_clicked(self._disable_erasing)

        pos[0] += BUTTON_WIDTH + BUTTON_SPACING
        ax_erase = self.fig.add_axes(pos)
        self.button_erase = Button(ax_erase, "", image=Image.open(ERASER_SYMBOL))
        self.button_erase.on_clicked(self._enable_erasing)

    def _on_key_press(self, event):
        if event.key == "ctrl+z" or event.key == "cmd+z":
            self._undo()
        elif event.key == "ctrl+y" or event.key == "cmd+y":
            self._redo()
        elif event.key == "e":
            self._enable_erasing()
        elif event.key == "d":
            self._disable_erasing()
        elif event.key == "v":
            self._toggle_visibility()

    def _enable_erasing(self, event=None):
        self._erasing = True

    def _disable_erasing(self, event=None):
        self._erasing = False

    def _toggle_visibility(self, event=None) -> None:
        self._visible = not self._visible
        if self._visible:
            self._ax_visibility.images[0].set_data(Image.open(VISIBILITY_ON_SYMBOL))
        else:
            self._ax_visibility.images[0].set_data(Image.open(VISIBILITY_OFF_SYMBOL))
        self._draw_mask()

    def _undo(self, event=None) -> None:
        if self._mask_history:
            self._mask_future.append(self._mask.copy())
            self.mask = self._mask_history.pop()
            self._draw_mask()

    def _redo(self, event=None) -> None:
        if self._mask_future:
            self._mask_history.append(self._mask.copy())
            self.mask = self._mask_future.pop()
            self._draw_mask()

    def _inverse_mask(self, event=None) -> None:
        self._mask_history.append(self._mask.copy())
        self._mask_future.clear()
        self.mask = np.logical_not(self.mask)
        self._draw_mask()

    def _draw_mask(self) -> None:
        if self._visible:
            self._mask_im.set_data(self._mask)
        else:
            self._mask_im.set_data(np.zeros_like(self._mask))
        self.fig.canvas.draw_idle()

    def _onselect(self, verts) -> None:
        self._mask_history.append(self._mask.copy())
        self._mask_future.clear()

        p = mpl.path.Path(verts)
        indices = p.contains_points(self.pix, radius=0).reshape(self._mask.shape)
        self._mask[indices] = not self._erasing
        self._draw_mask()

    def _ipython_display_(self) -> None:
        display(self.fig.canvas)  # type: ignore # noqa: F821

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @mask.setter
    def mask(self, val: np.ndarray) -> None:
        if val.shape != self._img.shape[:2]:
            msg = "Mask must have the same shape as the image"
            raise ValueError(msg)
        self._mask = val
