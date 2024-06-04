import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run


class GUI:
    def __init__(self, sim, bound=(-10, 10, -10, 10), d3=False):
        self.sim = sim
        self.bound = bound
        self.size = (bound[1] - bound[0], bound[3] - bound[2])
        self.d3 = d3
        self.init_scene()
        self.add_light()
        self.add_fps()
        self.init_sim()

    def init_scene(self):
        self.canvas = WgpuCanvas(max_fps=60)
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)

        self.scene = gfx.Scene()
        if self.d3:
            self.camera = gfx.PerspectiveCamera(50, 16/9)
            self.scene.add(self.camera.add(gfx.DirectionalLight()))
            self.controller = gfx.OrbitController(self.camera, register_events=self.renderer)
        else:
            self.camera = gfx.OrthographicCamera(*self.size)
            self.camera.show_rect(*self.bound)
        self.scene.add(gfx.Background(None, gfx.BackgroundMaterial("#ffffff")))

    def add_fps(self):
        self.stats = gfx.Stats(viewport=self.renderer)

    def add_light(self):
        self.scene.add(gfx.AmbientLight(intensity=0.8))

    def init_sim(self):
        w, h = self.canvas.get_logical_size()

        pos = self.sim.get_positions()

        if not self.d3:
            sizes = 2 * self.sim.get_radius() * min(w, h) / min(self.size)
            self.geometry = gfx.Geometry(positions=pos, sizes=sizes)
            material = gfx.PointsMaterial(color=(0, 1, 1, 1), size_mode="vertex", pick_write=True)
            points = gfx.Points(self.geometry, material)
            self.scene.add(points)
        else:
            sizes = self.sim.get_radius()
            self.spheres = []
            for coord, size in zip(pos, sizes):
                geometry = gfx.sphere_geometry(radius=size)
                material = gfx.MeshPhongMaterial(color=(0, 1, 1, 1))
                mesh = gfx.Mesh(geometry, material)
                mesh.local.position = coord
                self.spheres.append(mesh)
                self.scene.add(mesh)

            self.camera.show_object(self.scene)

            for wall in self.sim.get_walls():
                import numpy as np
                geometry = gfx.Geometry(positions=wall.astype(np.float32), colors=[[1, 0, 0, 1]])
                material = gfx.LineSegmentMaterial(thickness=6, color_mode="face")
                line = gfx.Line(geometry, material)
                self.scene.add(line)
    def animate(self):
        self.sim.step()
        if not self.d3:
            self.geometry.positions.data[:, :] = self.sim.get_positions()
            self.geometry.positions.update_range()
        else:
            for sphere, coord in zip(self.spheres, self.sim.get_positions()):
                sphere.local.position = coord
        if self.stats:
            with self.stats:
                self.renderer.render(self.scene, self.camera, flush=False)
                self.stats.render()
        self.canvas.request_draw()

    def run(self):
        self.canvas.request_draw(self.animate)
        run()