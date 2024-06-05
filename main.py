import numpy as np

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

rho = 2400 # kg/m^3
xmin = 0
xmax = 20
ymin = 0
ymax = 20
n = 10
np.random.seed(1)
positions = random_initial_conditions(n, 0.5, xmin, xmax, ymin, ymax)
velocities = np.random.uniform(-6,6,(n,2))
omega = np.random.uniform(-np.pi,np.pi,(n))
radius = np.random.uniform(0.5,0.5*0.5,(n))
dt = 1e-3
g = 9.81
sim = Simulation(positions, velocities,omega, radius,rho,g,d3=False)

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

gui = GUI(sim, bound=(0, 20, 0, 20), d3=False)

gui.run()