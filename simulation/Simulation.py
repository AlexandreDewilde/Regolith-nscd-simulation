from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from numba import jit

class Simulation:
    def __init__(self,
                init_positions: Annotated[list | NDArray, "2D or 3D list"],
                init_speeds: Annotated[list | NDArray, "2D or 3D list"],
                radius: Annotated[list | NDArray, "list of floats"],
                iM: Annotated[NDArray, "inverse mass"],
                lines: Annotated[list, "list of boundary lines"] = None,
                rectangles: Annotated[list, "list of boundary rectangles"] = None,
                kn = 2e7,
                gn = 10000,
                dt: Annotated[float, "time step"] = 0.01,
                d3: Annotated[bool, "3D or 2D simulation"] = False):
        """
        Args:
            init_positions: np.array of shape (n, 2|3) or list containing the initial positions of the particles
            init_speeds: np.array of shape (n, 2|3) or list containing the initial speeds of the particles
            radius: np.array of shape (n) or list containing the radius of the particles
        """
        if type(init_positions) == list:
            init_positions = np.array(init_positions)
        self.__positions = init_positions.astype(np.float64) if init_positions.shape[1] == 3 else np.c_[init_positions, np.zeros(len(init_positions))].astype(np.float64)

        if type(init_speeds) == list:
            init_speeds = np.array(init_speeds)
        self.__speeds = init_speeds.astype(np.float64) if init_speeds.shape[1] == 3 else np.c_[init_speeds, np.zeros(len(init_positions))].astype(np.float64)

        if type(radius) == list:
            radius = np.array(radius)
        self.__radius = radius.astype(np.float64)

        self.lines = np.array(lines).astype(np.float64) if lines is not None else np.array([[[0, 0, 0], [0, 0, 0]]]).astype(np.float64)
        self.rectangles = np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ]).astype(np.float64) if rectangles is None else np.array(rectangles).astype(np.float64)
        self.d3 = d3

        self.iM = iM
        self.kn = kn
        self.gn = gn

        self.old_contacts = {}
        self.t = 0
        self.dt = dt


    def add_grain(self, position: np.array,
                speed: NDArray,
                radius: float) -> None:
        """
        Add a grain to the simulation
        Args:
            position: np.array of shape (2|3) containing the position of the grain
            speed: np.array of shape (2|3) containing the speed of the grain
            radius: float containing the radius of the grain
        """
        self.__positions = np.vstack((self.__positions, position))
        self.__speeds = np.vstack((self.__speeds, speed))
        self.__radius = np.append(self.__radius, radius)

    def add_line(self, line: np.array) -> None:
        """
        Add a line to the simulation
        Args:
            line: np.array of shape (4, 2|3) containing the coordinates of the line
        """
        self.lines = np.vstack((self.lines, np.array([line])))

    def add_rectangle(self, rectangle: np.array) -> None:
        """
        Add a rectangle to the simulation
        Args:
            rectangle: np.array of shape (4, 2|3) containing the coordinates of the rectangle
        """
        self.rectangles = np.vstack((self.rectangles, np.array([rectangle])))

    def get_positions(self) -> np.array:
        return self.__positions.astype(np.float32)

    def get_radius(self) -> np.array:
        return self.__radius.astype(np.float32)

    def get_speeds(self) -> np.array:
        return self.__speeds.astype(np.float32)

    def get_lines(self) -> np.array:
        return self.lines.astype(np.float32)

    def get_rectangles(self) -> np.array:
        return self.rectangles.astype(np.float32)

    @staticmethod
    @jit(nopython=True)
    def solve_contacts(positions, speeds, radius, kn, gn):
        n = len(positions)
        f = np.zeros_like(positions).astype(np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                dst = np.linalg.norm(positions[i] - positions[j])
                if dst < radius[i] + radius[j]:
                    normal = (positions[i] - positions[j]) / dst
                    vi = speeds[i]
                    vj = speeds[j]
                    dv = vi - vj
                    vn = np.dot(dv, normal)
                    fn = kn * (radius[i] + radius[j] - dst) ** 1.5 - gn * vn
                    if fn > 0:
                        fi = fn * normal
                        f[i] += fi
                        f[j] -= fi
        return f

    @staticmethod
    @jit(nopython=True)
    def solve_contacts_line(positions, speeds, radius, lines, kn, gn):
        n = len(positions)
        f = np.zeros_like(positions).astype(np.float64)
        for i in range(n):
            for k in range(len(lines)):
                line = lines[k]
                t = line[1] - line[0]
                nt = np.linalg.norm(t)
                if np.abs(nt) < 1e-6:
                    continue
                xt = positions[i] - line[0]
                txt = np.dot(xt, t) / nt ** 2
                n = xt - txt * t
                dst = np.linalg.norm(n)
                if dst < radius[i]:
                    normal = n / dst
                    vi = speeds[i]
                    vn = np.dot(vi, normal)

                    fn = kn * (radius[i] - dst) ** 1.5 - vn * gn
                    if fn > 0:
                        f[i] += fn * normal
        return f

    @staticmethod
    @jit(nopython=True)
    def solve_contacts_rectangle(positions, speeds, radius, rectangles, kn, gn):
        f = np.zeros_like(positions).astype(np.float64)
        for k in range(len(rectangles)):
            rectangle = rectangles[k]
            ab, ac = rectangle[1] - rectangle[0], rectangle[2] - rectangle[0]
            n = np.cross(ab, ac)
            if np.linalg.norm(n) < 1e-6:
                continue
            n = n / np.linalg.norm(n)
            d = -np.dot(n, rectangle[0])
            for i in range(len(positions)):
                pos = positions[i]
                dst_plane_dir = (n[0] * pos[0] + n[1] * pos[1] + n[2] * pos[2] + d)
                dst_plane = np.abs(dst_plane_dir)
                h = pos - dst_plane_dir * n
                dst_h_rect = np.inf
                for j in range(4):
                    b, c = rectangle[j], rectangle[(j+1)%4]
                    vecd = (c - b) / np.linalg.norm(c - b)
                    v = h - b
                    t = np.dot(v, vecd)
                    p = b + t * vecd
                    dist = np.linalg.norm(p - h)
                    if dist < dst_h_rect:
                        dst_h_rect = dist

                dst = dst_plane + (dst_h_rect if not inside_rectangle(rectangle, h) else 0)
                if dst <= radius[i]:
                    normal = (pos - h) / np.linalg.norm(pos - h)
                    vi = speeds[i]
                    vn = np.dot(vi, normal)

                    fn = kn * (radius[i] - dst) ** 1.5 - vn * gn
                    if fn > 0:
                        f[i] += fn * normal
        return f

    def velocity_verlet(self) -> None:
        """
        Update the positions and speeds of the particles using the velocity verlet algorithm
        """
        f = self.solve_contacts(self.__positions, self.__speeds, self.__radius, self.kn, self.gn) + \
            self.solve_contacts_line(self.__positions, self.__speeds, self.__radius, self.lines, self.kn, self.gn) + \
            self.solve_contacts_rectangle(self.__positions, self.__speeds, self.__radius, self.rectangles, self.kn, self.gn)
        self.__speeds += self.dt * f * self.iM
        self.__positions += self.dt * self.__speeds
        # f = self.solve_contacts(self.__positions, self.__speeds, self.__radius, self.kn, self.gn) + self.solve_contacts_line(self.__positions, self.__speeds, self.__radius, self.lines, self.kn, self.gn)
        # self.__speeds += 0.5 * self.dt * f * self.iM

    def step(self) -> None:
        """
        Perform a simulation step
        """
        self.velocity_verlet()
        self.t += self.dt

@jit(nopython=True)
def inside_rectangle(rectangle, point):

    mn_x = min(rectangle[0][0], rectangle[1][0], rectangle[2][0], rectangle[3][0])
    mx_x = max(rectangle[0][0], rectangle[1][0], rectangle[2][0], rectangle[3][0])
    mn_y = min(rectangle[0][1], rectangle[1][1], rectangle[2][1], rectangle[3][1])
    mx_y = max(rectangle[0][1], rectangle[1][1], rectangle[2][1], rectangle[3][1])
    mn_z = min(rectangle[0][2], rectangle[1][2], rectangle[2][2], rectangle[3][2])
    mx_z = max(rectangle[0][2], rectangle[1][2], rectangle[2][2], rectangle[3][2])
    return mn_y <= point[1] <= mx_y and mn_z <= point[2] <= mx_z and mn_x <= point[0] <= mx_x