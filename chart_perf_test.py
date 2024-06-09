import numpy as np

from gui.GUI import GUI
from simulation.Simulation import Simulation

def insert_particles(N, R, xmin, xmax, ymin, ymax):
    # Calculate the box dimensions
    width = xmax - xmin
    height = ymax - ymin

    # Calculate the diameter of the particles
    diameter = 2 * R

    # Calculate the number of particles that can fit along each dimension
    num_particles_x = int(width // diameter)
    num_particles_y = int(height // diameter)

    # Adjust the number of particles if needed
    total_particles = min(N, num_particles_x * num_particles_y)

    # Create a list to store particle positions
    particles = []

    # Calculate the spacing between particles
    x_spacing = width / num_particles_x
    y_spacing = height / num_particles_y

    # Place the particles in a grid
    count = 0
    for i in range(num_particles_x):
        for j in range(num_particles_y):
            if count >= total_particles:
                break
            x = xmin + (i + 0.5) * x_spacing
            y = ymin + (j + 0.5) * y_spacing
            particles.append((x, y))
            count += 1
        if count >= total_particles:
            break

    return np.array(particles)

rho = 2400 # kg/m^3
xmin = 0
xmax = 20
ymin = 0
ymax = 20
n = 10
np.random.seed(1)

dt = 1e-6
g = 9.81


for i in range(100, 2001, 100):
    print(f"test {i}")
    positions = insert_particles(i, 1/4, xmin, xmax, ymin, ymax)
    n = len(positions)
    velocities = np.zeros((n, 2))
    omega = np.zeros(len(positions))
    radius = np.ones(len(positions)) / 4
    for tree in [False, True]:
        sim = Simulation(
            positions,
            velocities,
            omega,
            radius,
            rho,
            g,
            tend=0.1,
            tree=tree,
            test_perf=True,
        )

        sim.add_line(
            np.array([
                [xmin, ymax, 0],
                [xmax, ymax, 0],
            ]).astype(np.float64)
        )
        sim.add_line(
            np.array([
                [xmax, ymin, 0],
                [xmax, ymax, 0],
            ]).astype(np.float64)
        )
        sim.add_line(
            np.array([
                [xmin, ymin, 0],
                [xmin, ymax, 0],
            ]).astype(np.float64)
        )
        sim.add_line(
            np.array([
                [xmin, ymin, 0],
                [xmax, ymin, 0],
            ]).astype(np.float64)
        )


        sim.run_sim()