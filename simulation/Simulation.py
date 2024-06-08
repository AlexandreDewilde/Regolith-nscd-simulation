import signal
import sys

from typing import Annotated
import numpy as np
from numpy.typing import NDArray
from .tree import *
import trimesh
import time
from .tree import *

from .contact import solve_contacts_jacobi, detect_contacts


class Simulation:
    def __init__(self,
                init_positions: Annotated[list | NDArray, "2D or 3D list"],
                init_velocities: Annotated[list | NDArray, "2D or 3D list"],
                init_omega: Annotated[list | NDArray, "2D or 3D list"],
                radius: Annotated[list | NDArray, "list of floats"],
                rho: Annotated[float, "Density [kg/m^3]"],
                g: Annotated[float, "Gravity [m/s^2]"] = 9.81,
                mu: Annotated[float, "Friction coefficient"] = 0.3,
                tree : Annotated[bool, "Use of QuadTree"] = True,
                lines: Annotated[list, "list of boundary lines"] = None,
                rectangles: Annotated[list, "list of boundary rectangles"] = None,
                meshes: Annotated[list, "list of meshes"] = None,
                dt: Annotated[float, "time step"] = 0.01,
                d3: Annotated[bool, "3D or 2D simulation"] = False,
                precomputation_file: Annotated[str, "file containing the precomputation data, if no file no precompute"] = None,
                tend: Annotated[float, "end time"] = None,
        ):
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
        self.n_lines = len(lines) if lines is not None else 0

        self.rectangles = np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ]).astype(np.float64) if rectangles is None else np.array(rectangles).astype(np.float64)


        self.d3 = d3

        self.t = 0
        self.tend = tend
        self.dt = dt

        self.iM = (1 / (np.pi * rho * self.__radius ** 2))
        self.I = 1 / 2 * 1 / self.iM * self.__radius * self.__radius

        self.contacts = []
        self.old_contacts = []

        self.g = np.array([[0, -g, 0]for i in range(len(self.__velocities))])

        self.mu = mu

        self.meshes = meshes if meshes is not None else []
        self.meshes_positions = [np.ascontiguousarray(mesh.vertices, dtype=np.float64) for mesh in self.meshes]
        self.meshes_faces = [np.ascontiguousarray(mesh.faces, dtype=np.int32) for mesh in self.meshes]

        self.precomputation_file = precomputation_file

        self.tree = tree

        self.t_history = [self.t]
        self.positions_history = [self.__positions]
        self.velocities_history = [self.__velocities]
        self.omega_history = [self.__omega]

        signal.signal(signal.SIGINT, self.signal_handler)


    def signal_handler(self, sig, frame):
        self.write_precomputation()
        sys.exit(0)

    def run_sim(self):
        while self.tend is None or self.t < self.tend:
            self.step()
            self.t_history.append(self.t)
            self.positions_history.append(self.__positions)
            self.velocities_history.append(self.__velocities)
            self.omega_history.append(self.__omega)
        self.write_precomputation()

    def write_precomputation(self):
        if self.precomputation_file is not None:
            with open(self.precomputation_file, "w") as f:
                for t, positions, velocities, omega in zip(self.t_history, self.positions_history, self.velocities_history, self.omega_history):
                    f.write(f"Computation time: {t}\n")
                    for i, (pos, vel, om) in enumerate(zip(positions, velocities, omega)):
                        f.write(f"\tPosition {i}: {pos[0]} {pos[1]} {pos[2]}\n")
                        f.write(f"\tVelocity {i}: {vel[0]} {vel[1]} {vel[2]}\n")
                        f.write(f"\tOmega {i}: {om}\n")
                    f.write("\n")

    def add_mesh(self, mesh, scale, position) -> None:
        mesh.apply_transform(trimesh.transformations.scale_and_translate(scale, -mesh.centroid * scale + position))
        self.meshes.append(mesh)

        self.meshes_positions.append(np.ascontiguousarray(mesh.vertices, dtype=np.float64))
        self.meshes_faces.append(np.ascontiguousarray(mesh.faces, dtype=np.int32))

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
        if self.n_lines == 0:
            self.lines = np.array([line])
            self.n_lines += 1
        else:
            self.lines = np.vstack((self.lines, np.array([line])))
            self.n_lines += 1


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

    def get_meshes(self) -> list[trimesh.Trimesh]:
        return self.meshes

    def step(self) -> None:
        """
        Update the positions and velocities of the particles
        """
        tic = time.time()
        #self.__velocities += self.g*self.dt
        if self.tree == True:
            tree = set_tree(self.__positions)
            print("Tree time : ",time.time()-tic)
            tic = time.time()
            self.contacts = detect_contacts(self.__positions, self.__velocities, self.__radius, self.lines, self.dt,tree)
            print("Detection time : ",time.time()-tic)

        else :
            self.contacts = detect_contacts(self.__positions, self.__velocities, self.__radius, self.lines, self.dt,None)
            print("Detection time : ",time.time()-tic)

        tic = time.time()
        self.__velocities,self.__omega = solve_contacts_jacobi(self.contacts, self.__positions, self.__velocities, self.__omega, self.__radius, self.iM, self.I, self.dt)
        print("Solving time : ",time.time()-tic)
        self.__positions += self.__velocities * self.dt
        self.t += self.dt