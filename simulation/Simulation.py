from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from numba import jit

from .contact import solve_contacts, detect_contacts


class Simulation:
    def __init__(self,
                init_positions: Annotated[list | NDArray, "2D or 3D list"],
                init_velocities: Annotated[list | NDArray, "2D or 3D list"],
                init_omega: Annotated[list | NDArray, "2D or 3D list"],
                radius: Annotated[list | NDArray, "list of floats"],
                rho: Annotated[float, "Density [kg/m^3]"],
                g: Annotated[float, "Gravity [m/s^2]"] = 9.81,
                mu: Annotated[float, "Friction coefficient"] = 0.3,
                lines: Annotated[list, "list of boundary lines"] = None,
                rectangles: Annotated[list, "list of boundary rectangles"] = None,
                restitution_wall: Annotated[float, "restitution coefficient"] = 0,
                restitution_particles: Annotated[float, "restitution coefficient"] = 0,
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

        if type(init_velocities) == list:
            init_velocities = np.array(init_velocities)
        self.__velocities = init_velocities.astype(np.float64) if init_velocities.shape[1] == 3 else np.c_[init_velocities, np.zeros(len(init_positions))].astype(np.float64)

        if type(radius) == list:
            radius = np.array(radius)
        self.__radius = radius.astype(np.float64)

        if type(init_omega) == list:
            init_omega = np.array(init_omega)
        self.__omega = init_omega.astype(np.float64)

        self.lines = np.array(lines).astype(np.float64) if lines is not None  else np.array([[[0, 0, 0], [0, 0, 0]]]).astype(np.float64)

        self.rectangles = np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ]).astype(np.float64) if rectangles is None else np.array(rectangles).astype(np.float64)


        self.d3 = d3

        self.t = 0
        self.dt = dt

        self.iM = (1 / (np.pi * rho * self.__radius ** 2))
        self.I = 1 / 2 * 1 / self.iM * self.__radius * self.__radius

        self.contacts = []
        self.old_contacts = []

        self.g = np.array([[0, -g, 0]for i in range(len(self.__velocities))])

        self.restitution_wall = restitution_wall
        self.restitution_particles = restitution_particles
        self.mu = mu


    def add_grain(self,
                position: NDArray,
                velocity: NDArray,
                radius: float) -> None:
        """
        Add a grain to the simulation
        Args:
            position: np.array of shape (2|3) containing the position of the grain
            speed: np.array of shape (2|3) containing the speed of the grain
            radius: float containing the radius of the grain
        """
        self.__positions = np.vstack((self.__positions, position))
        self.__velocities = np.vstack((self.__velocities, velocity))
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

    def get_velocities(self) -> np.array:
        return self.__velocities.astype(np.float32)

    def get_lines(self) -> np.array:
        return self.lines.astype(np.float32)

    def get_rectangles(self) -> np.array:
        return self.rectangles.astype(np.float32)

    @staticmethod
    @jit(nopython=True)
    def generate_tree(positions):
        #tree = QuadTree()
        return

    @staticmethod
    @jit(nopython=True)
    def jacobi(contacts, positions, velocities, omega, radius, iM, I, restitution_wall, dt):
        while 1:
            if contacts[0].type == -1 :
                break
            impulses = solve_contacts(positions, velocities, omega, radius, iM,
                                            I, contacts, restitution_wall, dt)
            if np.sum(impulses) < 1e-3 / len(radius):
                break

    def step(self) -> None:
        """
        Update the positions and speeds of the particles using the velocity verlet algorithm
        """
        self.generate_tree(self.__positions)
        self.__velocities += self.g * self.dt
        self.contacts = detect_contacts(self.__positions, self.__velocities, self.__radius, self.lines, self.rectangles, self.dt)
        self.jacobi(self.contacts, self.__positions, self.__velocities, self.__omega, self.__radius, self.iM, self.I, self.restitution_wall, self.dt)
        self.__positions += self.__velocities * self.dt
        self.t += self.dt
