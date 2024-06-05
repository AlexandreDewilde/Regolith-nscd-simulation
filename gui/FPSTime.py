import pygfx as gfx
import time

from pygfx import Stats

class FPSTime(Stats):

    def __init__(self, sim_time=0, *args, **kwargs):
        self.sim_time = sim_time
        super().__init__(*args, **kwargs)
        self.bg.local.scale = (130, self._line_height * 3.1, 1)

    def set_sim_time(self, sim_time):
        self.sim_time = sim_time

    def stop(self):
        super().stop()
        if not self._init:
            self._init = True
            return

        t = time.perf_counter_ns()
        self._frames += 1

        delta = round((t - self._tbegin) / 1_000_000)
        self._tmin = min(self._tmin, delta)
        self._tmax = max(self._tmax, delta)

        if t >= self._tprev + 1_000_000_000:
            # update FPS counter whenever a second has passed
            fps = round(self._frames / ((t - self._tprev) / 1_000_000_000))
            self._tprev = t
            self._frames = 0
            self._fmin = min(self._fmin, fps)
            self._fmax = max(self._fmax, fps)
            self._fps = fps

        text = f"{delta} ms ({self._tmin}-{self._tmax})"
        if self._fps is not None:
            text += f"\n{self._fps} fps ({self._fmin}-{self._fmax})"
        text += f"\nSimulation time: {self.sim_time:.2f} s"
        self.stats_text.geometry.set_text(text)
