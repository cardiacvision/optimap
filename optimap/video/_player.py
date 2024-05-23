import matplotlib.widgets
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


class InteractivePlayer(FuncAnimation):
    """Generic interactive looping player with play, pause, forward, backward, and slider controls based on :class:`matplotlib.animation.FuncAnimation`.

    Calls a function at each frame to update the figure content with the current frame index. Can be used to create custom video players or other interactive animations, see :func:`optimap.show_video` as an example of how to use this class.

    .. table:: **Keyboard Shortcuts**

        ========================= ===========================
        Key                       Action
        ========================= ===========================
        ``Space``                 Play/Pause
        ``Left``                  Last frame
        ``Right``                 Next frame
        ``Up``                    Increase step size
        ``Down``                  Decrease step size
        ``q``                     Quit
        ========================= ===========================


    Example
    -------

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from optimap.video import InteractivePlayer
        fig, ax = plt.subplots()
        x = np.linspace(0, 2*np.pi, 100)
        line, = ax.plot(x, np.sin(x))
        def update(i):
            line.set_ydata(np.sin(x + i / 10))
        player = InteractivePlayer(fig, update, end=len(x))
        plt.show()

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to plot on.
    func : callable
        Function to call at each frame. Must accept a single integer argument (frame index).
    start : int, optional
        Start frame, by default 0
    end : int, optional
        End frame (exclusive), by default 100
    step : int, optional
        Step size for forward and backward buttons, by default 1
    interval : int, optional
        Delay between frames in milliseconds, by default 25
    gui_pos : tuple, optional
        Position of the GUI elements (play, slider, etc.), by default (0.125, 0.92)
    **kwargs
        Additional keyword arguments passed to :class:`matplotlib.animation.FuncAnimation`
    """

    def __init__(
        self,
        fig,
        func,
        start=0,
        end=100,
        step=1,
        interval=25,
        gui_pos=(0.125, 0.92),
        **kwargs,
    ):
        self.i = start - step
        self.min = start
        self.max = end
        self.runs = True
        self.forwards = True
        self.step = step
        self.fig = fig
        self.func = func
        self.saving = False
        self.suptitle = fig.suptitle("  ", font="monospace")
        fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.setup_gui(gui_pos)
        FuncAnimation.__init__(
            self,
            self.fig,
            self.update,
            frames=self.play(),
            cache_frame_data=False,
            repeat=False,
            interval=interval,
            **kwargs,
        )

    def play(self):
        while self.runs:
            self.i = self.i + self.step
            if self.i >= self.min and self.i < self.max:
                yield self.i
            else:
                if self.saving:
                    break
                elif self.i >= self.max:
                    yield self.min
                else:
                    yield self.min

    def update(self, i):
        self.slider.set_val(i)
        if self.saving:
            self.suptitle.set_text(f"Frame {i:4d}")

    def set_pos(self, i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def toggle_play(self, event=None):
        if self.runs:
            self.runs = False
            self.event_source.stop()
            self.button_stop.label.set_text("▶")
        else:
            self.runs = True
            self.event_source.start()
            self.button_stop.label.set_text("■")
        self.fig.canvas.draw_idle()

    def one_forward(self, event=None):
        self.onestep(True)

    def one_backward(self, event=None):
        self.onestep(False)

    def onestep(self, forwards):
        x = 2 * forwards - 1
        self.step = abs(self.step) * x
        self.i += x
        if self.i >= self.max:
            self.i = self.min
        elif self.i < self.min:
            self.i = self.max - 1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == " ":
            self.toggle_play()
        elif event.key == "right":
            self.one_forward()
        elif event.key == "left":
            self.one_backward()
        elif event.key == "up":
            self.step += 1
        elif event.key == "down":
            self.step -= 1

    def setup_gui(self, pos):
        self.ax_player = self.fig.add_axes([pos[0], pos[1], 0.64, 0.04])
        divider = make_axes_locatable(self.ax_player)
        self.ax_slider = divider.append_axes("right", size="500%", pad=0.07)
        self.button_stop = matplotlib.widgets.Button(self.ax_player, label="■")
        self.button_stop.on_clicked(self.toggle_play)
        self.slider = matplotlib.widgets.Slider(
            self.ax_slider, "", self.min, self.max - 1, valinit=self.min,
        )
        self.slider.on_changed(self.set_pos)

    def save(self, *args, hide_slider=True, hide_buttons=True, hide_framecounter=False, **kwargs):
        """Save the animation as a movie file, but hide the GUI buttons and slider.

        Parameters
        ----------
        *args :
            See FuncAnimation's :meth:`~matplotlib.animation.Animation.save` for arguments.
        hide_slider : bool, optional
            Hide the slider, by default True
        hide_buttons : bool, optional
            Hide the play/pause button, by default True
        hide_framecounter : bool, optional
            Hide the frame counter, by default False
        **kwargs :
            See FuncAnimation's :meth:`~matplotlib.animation.Animation.save` for arguments.
        """
        self.i = self.min - self.step
        self.saving = True
        self._save_count = (self.max - self.min) // self.step
        self.ax_player.set_visible(not hide_buttons)
        self.ax_slider.set_visible(not hide_slider)
        self.suptitle.set_visible(not hide_framecounter)

        super().save(*args, **kwargs)
        
        self.saving = False
        # reset frame generator, it's exhausted now
        self._iter_gen = self.play
        self._save_count = None

        self.ax_player.set_visible(True)
        self.ax_slider.set_visible(True)
        self.suptitle.set_visible(True)
