import numpy as np
import trimesh

from gui.GUI import GUI
from simulation.Simulation import Simulation

def random_initial_conditions(n, R, xmin, xmax, ymin, ymax):
  pos = np.zeros((n,2))
  for i in range(n):
    xx = np.random.uniform(xmin+R,xmax-R) #random x position inside [xmin,xmax]
    yy = np.random.uniform(ymin+R,ymax-R) #random y position inside [ymin,ymax]
    while any((pos[:,0]-xx)**2 + (pos[:,1]-yy)**2 < 4*R**2): #while the random position intersects an already existing grain, compute a new one
      xx = np.random.uniform(xmin+R,xmax-R)
      yy = np.random.uniform(ymin+R,ymax-R)
    pos[i,:] = [xx,yy]

  return pos

def open_mesh_files(*mesh_files):
    meshes = []
    for file in mesh_files:
        s = trimesh.load_mesh(file)
        # for name, mesh in s.geometry.items():
            # mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        meshes.append(s)
    return meshes

rho = 2400 # kg/m^3
xmin = 0
xmax = 20
ymin = 0
ymax = 20
zmin = 0
zmax = 20
n = 10
np.random.seed(1)
positions = random_initial_conditions(n, 0.5, xmin, xmax, ymin, ymax)
positions = np.c_[positions, np.ones(len(positions)) * 10]
velocities = np.random.uniform(-6,6,(n,2))
omega = np.random.uniform(-np.pi,np.pi,(n))
radius = np.random.uniform(0.5,0.5*0.5,(n))
dt = 1e-3
g = 9.81

meshes = open_mesh_files("mesh/cube.obj")
sim = Simulation(positions, velocities,omega, radius, rho, g, d3=True)
sim.add_mesh(meshes[0], 0.5, np.array([10, 0, 10]))

sim.add_rectangle(
    np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin]
    ])
)
sim.add_rectangle(
    np.array([
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax]
    ])
)
sim.add_rectangle(
    np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmin, ymin, zmax]
    ])
)
sim.add_rectangle(
    np.array([
        [xmin, ymax, zmin],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax]
    ])
)

sim.add_rectangle(
    np.array([
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
        [xmax, ymin, zmax]
    ])
)

sim.add_rectangle(
    np.array([
        [xmin, ymin, zmin],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmin, ymin, zmax]
    ])
)

gui = GUI(sim, bound=(0, 20, 0, 20), d3=True)

gui.run()