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
n = 100
np.random.seed(1)
positions = insert_particles(n, 1, xmin, xmax, ymin, ymax)
n = len(positions)
print(n)
velocities = np.random.uniform(-10,10,(n,2))
#velocities = np.array([[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0]])
omega = np.zeros(len(positions))
radius =np.ones(len(positions))
dt = 1e-6
g = 9.81
sim = Simulation(
    positions,
    velocities,omega,
    radius,
    rho,
    g,
    d3=False,
    tree=False,
    precomputation_file="out.txt"
)

sim.add_line(
    np.array([
        [xmin, ymax, 0],
        [xmax, ymax, 0],
    ])
)
sim.add_line(
    np.array([
        [xmax, ymin, 0],
        [xmax, ymax, 0],
    ])
)
sim.add_line(
    np.array([
        [xmin, ymin, 0],
        [xmin, ymax, 0],
    ])
)
sim.add_line(
    np.array([
        [xmin, ymin, 0],
        [xmax, ymin, 0],
    ])
)

sim.run_sim()
exit()
gui = GUI(sim, bound=(0, 20, 0, 20), d3=False)

gui.run()