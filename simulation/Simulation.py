import signal
import sys
import time

import trimesh
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from .contact import solve_contacts_jacobi, detect_contacts


class Simulation:
    def __init__(self,
            init_positions: list | NDArray,
            init_velocities: list | NDArray,
            init_omega: list | NDArray,
            radius: list | NDArray,

            rho: float,
            g: float = 9.81,
            mu: float = 0.3,

            lines: list = None,
            rectangles: list = None,
            meshes: list = None,

            dt: float = 0.01,
            tend: float = None,

            tree : bool = True,
            d3: bool = False,
            precomputation_file: str = None,
            debug: bool = False
        ) -> None:
        """
        Args:
            init_positions: np.array of shape (n, 2|3) or list containing the initial positions of the particles
            init_speeds: np.array of shape (n, 2|3) or list containing the initial speeds of the particles
            init_omega: np.array of shape (n) or list containing the initial angular speeds of the particles
            radius: np.array of shape (n) or list containing the radius of the particles

            rho: float containing the density of the particles
            g: float containing the gravity
            mu: float containing the friction coefficient

            lines: list of np.array of shape (2, 2|3) containing the coordinates of the boundary lines
            rectangles: list of np.array of shape (4, 2|3) containing the coordinates of the rectangles
            meshes: list of trimesh.Trimesh containing the meshes

            dt: float containing the time step
            tend: float containing the end time

            tree: bool to use the tree or not
            d3: bool to use 3D or 2D simulation
            precomputation_file: str containing the file to write the precomputation data
            debug: bool to print debug information
        """

        self.__init_particles(init_positions, init_velocities, init_omega, radius)
        self.contacts = []

        self.__init_physical_properties(rho, g, mu)

        self.__init_boundary(lines, rectangles, meshes)

        self.t = 0
        self.tend = tend
        self.dt = dt
        self.__init_particles_history()

        self.d3 = d3
        self.precomputation_file = precomputation_file
        self.tree = tree
        self.debug = debug

        # When the user presses Ctrl+C, the program will write the precomputation data to the file
        signal.signal(signal.SIGINT, self.__signal_handler)

    def __init_particles(self,
            init_positions: list | NDArray,
            init_velocities: list | NDArray,
            init_omega: list | NDArray,
            radius: list | NDArray,
        ) -> None:
        """
        Initialize the particles and convert the lists to np.array when needed
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

    def __init_particles_history(self) -> None:
        """
        Initialize the history of the particles
        """
        self.t_history = [self.t]
        self.positions_history = [self.__positions]
        self.velocities_history = [self.__velocities]
        self.omega_history = [self.__omega]

    def __init_physical_properties(self, rho: float, g: float, mu: float) -> None:
        self.g = np.array([[0, -g, 0] for _ in range(len(self.__velocities))])
        self.mu = mu
        self.rho = rho
        self.iM = (1 / (np.pi * self.rho * self.__radius ** 2))
        self.I = 1 / 2 * 1 / self.iM * self.__radius * self.__radius

    def __init_boundary(self, lines, rectangles, meshes) -> None:
        self.lines = np.array(lines).astype(np.float64) if lines is not None  else np.array([[[0, 0, 0], [0, 0, 0]]]).astype(np.float64)
        self.n_lines = len(lines) if lines is not None else 0
        self.rectangles = np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ]).astype(np.float64) if rectangles is None else np.array(rectangles).astype(np.float64)
        self.meshes = meshes if meshes is not None else []

        # Following two lists are needed for contact detection
        self.meshes_positions = [np.ascontiguousarray(mesh.vertices, dtype=np.float64) for mesh in self.meshes]
        self.meshes_faces = [np.ascontiguousarray(mesh.faces, dtype=np.int32) for mesh in self.meshes]

    def __signal_handler(self, sig, frame) -> None:
        self.__write_precomputation()
        sys.exit(0)

    def run_sim(self) -> None:
        """
        Run the simulation until the end time
        """
        while self.tend is None or self.t < self.tend:
            self.step()
            self.t_history.append(self.t)
            self.positions_history.append(self.__positions)
            self.velocities_history.append(self.__velocities)
            self.omega_history.append(self.__omega)
        self.__write_precomputation()

    def __write_precomputation(self) -> None:
        if self.precomputation_file is not None:
            with open(self.precomputation_file, "w") as f:
                for t, positions, velocities, omega in zip(self.t_history, self.positions_history, self.velocities_history, self.omega_history):
                    f.write(f"Computation time: {t}\n")
                    for i, (pos, vel, om) in enumerate(zip(positions, velocities, omega)):
                        f.write(f"\tPosition {i}: {pos[0]} {pos[1]} {pos[2]}\n")
                        f.write(f"\tVelocity {i}: {vel[0]} {vel[1]} {vel[2]}\n")
                        f.write(f"\tOmega {i}: {om}\n")
                    f.write("\n")

    def add_mesh(self, mesh: trimesh.Trimesh, scale: float, position: NDArray) -> None:
        """
        Add a mesh to the simulation
        Args:
            mesh: trimesh.Trimesh containing the mesh
            scale: float containing the scale of the mesh
            position: np.array of shape (2|3) containing the position of the mesh
        """
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

    def compute_cohesion_force(self) -> np.array:
        forces = np.zeros_like(self.__positions)

        for i in range(self.__positions.shape[0]):
            for j in range(i + 1, self.__positions.shape[0]):
                direction = self.__positions[j] - self.__positions[i]
                distance = np.linalg.norm(direction)
                force_magnitude = 10000.0
                force = force_magnitude * direction / distance  # Normalize the direction vector
                forces[i] += force
                forces[j] -= force  # Newton's third law

        return forces

    def step(self) -> None:
        """
        Update the positions and velocities of the particles
        """
        forces = 1/self.iM[:,np.newaxis]*self.g \
            #+ self.compute_cohesion_force()

        self.__velocities = self.dt * forces * self.iM[:, np.newaxis] + self.__velocities

        if self.tree:
            self.__detect_contacts_tree()
        else:
            self.__detect_contacts()

        tic = time.time()
        self.__velocities,self.__omega = solve_contacts_jacobi(
            self.contacts,
            self.__positions,
            self.__velocities,
            self.__omega,
            self.__radius,
            self.iM,
            self.I,
            self.dt
        )

        if self.debug:
            print("Solving time : ",time.time() - tic)

        self.__positions += self.__velocities * self.dt
        self.t += self.dt

    def __detect_contacts_tree(self, nsub: int):
        tic = time.time()

        tree = KDTree(self.__positions)
        IDs = tree.query_ball_point(self.__positions, np.max(self.__radius) * 3)
        new_IDs = []
        for i in range(len(IDs)):
            l = np.array([IDs[i][j] for j in range(len(IDs[i]))])
            new_IDs.append(l)

        if self.debug:
            print("Tree time : ",time.time() - tic)
        tic = time.time()

        self.contacts = detect_contacts(self.__positions, self.__velocities, self.__radius, self.lines, self.dt, new_IDs)

        if self.debug:
            print("Detection time : ",time.time() - tic)

    def __detect_contacts(self):
        tic = time.time()

        self.contacts = detect_contacts(self.__positions, self.__velocities, self.__radius, self.lines, self.dt, None)

        if self.debug:
            print("Detection time : ",time.time() - tic)