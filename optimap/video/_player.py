from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.widgets


class Player(FuncAnimation):
    def __init__(
        self,
        fig,
        func,
        frames=None,
        init_func=None,
        fargs=None,
        save_count=None,
        mini=0,
        maxi=100,
        pos=(0.125, 0.92),
        step=1,
        **kwargs
    ):
        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.step = step
        self.fig = fig
        self.func = func
        self.saving = False
        self.suptitle = fig.suptitle(f"  ", font="monospace")
        fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.setup(pos)
        FuncAnimation.__init__(
            self,
            self.fig,
            self.update,
            frames=self.play(),
            init_func=init_func,
            fargs=fargs,
            save_count=save_count,
            cache_frame_data=False,
            repeat=False,
            **kwargs
        )

    def play(self):
        while self.runs:
            self.i = self.i + self.step * (self.forwards - (not self.forwards))
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                if self.saving:
                    break
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

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == "right":
            self.oneforward()
        elif event.key == "left":
            self.onebackward()
        elif event.key == " ":
            self.toggle_play()

    def setup(self, pos):
        self.ax_player = self.fig.add_axes([pos[0], pos[1], 0.64, 0.04])
        divider = make_axes_locatable(self.ax_player)
        self.ax_slider = divider.append_axes("right", size="500%", pad=0.07)
        self.button_stop = matplotlib.widgets.Button(self.ax_player, label="■")
        self.button_stop.on_clicked(self.toggle_play)
        self.slider = matplotlib.widgets.Slider(
            self.ax_slider, "", self.min, self.max, valinit=self.i
        )
        self.slider.on_changed(self.set_pos)

    def save(self, *args, **kwargs):
        self.saving = True
        # self.save_count = self.max // self.step

        self.ax_player.set_visible(False)
        self.ax_slider.set_visible(False)
        super().save(*args, **kwargs)
        
        self.saving = False
        # self.save_count = None
        self.ax_player.set_visible(True)
        self.ax_slider.set_visible(True)