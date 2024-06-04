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


positions = random_initial_conditions(100, 0.5, 0, 20, 0, 20)
speeds = np.random.uniform(-6,6,(100,2))
radius = np.random.uniform(0.5,0.5*0.5,(100,1))
dt = 0.5*(4*0.5*0.5**2.5*np.pi*3/(3*1e4))**0.5

sim = Simulation(positions, speeds, radius, d3=False)
sim.add_wall(np.array([[0, 0, 0],[20, 0, 0]]))
sim.add_wall(np.array([[20, 0, 0],[20,20, 0]]))
sim.add_wall(np.array([[0, 20, 0],[20, 20, 0]]))
sim.add_wall(np.array([[0, 0, 0],[0, 20, 0]]))
gui = GUI(sim, bound=(0, 20, 0, 20), d3=True)

gui.run()