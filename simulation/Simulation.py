from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from numba import jit

class Simulation:
    def __init__(self,
                init_positions: Annotated[list | NDArray, "2D or 3D list"],
                init_speeds: Annotated[list | NDArray, "2D or 3D list"],
                radius: Annotated[list | NDArray, "list of floats"],
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
        self.__positions = init_positions.astype(np.float32) if init_positions.shape[1] == 3 else np.c_[init_positions, np.zeros(len(init_positions))].astype(np.float32)

        if type(init_speeds) == list:
            init_speeds = np.array(init_speeds)
        self.__speeds = init_speeds.astype(np.float32) if init_speeds.shape[1] == 3 else np.c_[init_speeds, np.zeros(len(init_positions))].astype(np.float32)

        if type(radius) == list:
            radius = np.array(radius)
        self.__radius = radius.astype(np.float32)

        self.d3 = d3
        self.iM = np.ones((len(self.__positions), 3))

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

    def get_positions(self) -> np.array:
        return self.__positions

    def get_radius(self) -> np.array:
        return self.__radius

    def get_speeds(self) -> np.array:
        return self.__speeds

    @staticmethod
    @jit(nopython=True)
    def solve_contacts(positions, speeds, radius):
        n = len(positions)
        f = np.zeros_like(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dst = np.linalg.norm(positions[i] - positions[j])
                if dst < radius[i] + radius[j]:
                    normal = (positions[i] - positions[j]) / dst
                    vi = speeds[i]
                    vj = speeds[j]
                    dv = vi - vj
                    vn = np.dot(dv, normal)
                    kn = gn = 1.0
                    fn = kn * np.abs(dst - radius[i] - radius[j]) ** 1.5 - gn*vn
                    if fn > 0:
                        fi = fn * normal
                        f[i] += fi
                        f[j] -= fi
        return f


    def velocity_verlet(self) -> None:
        """
        Update the positions and speeds of the particles using the velocity verlet algorithm
        """
        f = self.solve_contacts(self.__positions, self.__speeds, self.__radius)
        self.__speeds += 0.5 * self.dt * f * self.iM
        self.__positions += self.dt * self.__speeds
        f = self.solve_contacts(self.__positions, self.__speeds, self.__radius)
        self.__positions += 0.5 * self.dt * f * self.iM

    def step(self) -> None:
        """
        Perform a simulation step
        """
        self.velocity_verlet()
        self.t += self.dt