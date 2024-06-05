from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from numba import jit
from numba.typed import List
from numba.experimental import jitclass
from numba import int32, float32    # import the types

class Simulation:
    def __init__(self,
                init_positions: Annotated[list | NDArray, "2D or 3D list"],
                init_velocities: Annotated[list | NDArray, "2D or 3D list"],
                init_omega: Annotated[list | NDArray, "2D or 3D list"],
                radius: Annotated[list | NDArray, "list of floats"],
                rho: Annotated[float, "Density [kg/m^3]"],
                g:Annotated[float, "Gravity [m/s^2]"],
                lines: Annotated[list, "list of boundary lines"] = None,
                rectangles: Annotated[list, "list of boundary rectangles"] = None,
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
            omega = np.array(init_omega)
        self.__omega = init_omega.astype(np.float64)  
        self.lines = np.array(lines).astype(np.float64) if lines is not None  else np.array([[[0, 0, 0], [0, 0, 0]]]).astype(np.float64)
        self.n_lines = len(lines) if lines is not None else 0
        self.rectangles = np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ]).astype(np.float64) if rectangles is None else np.array(rectangles).astype(np.float64)
        self.d3 = d3
        self.iM = (1/(np.pi*rho*self.__radius**2))
        self.I = 1/2*1/self.iM*self.__radius*self.__radius
        self.old_contacts = []
        self.t = 0
        self.dt = dt
        self.contacts = []
        self.g = np.array([[0,-g,0]for i in range(len(self.__velocities))])
        self.restitution_wall = 0
        self.restitution_particles = 0
        self.mu = 0.3


    def add_grain(self, position: np.array,
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
        else : 
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

    @staticmethod
    @jit(nopython=True) 
    def generate_tree(positions):
        #tree = QuadTree()
        return

    @staticmethod
    @jit(nopython=True)            
    def detect_contacts(positions,velocities,radius,walls,dt):
        detection_range = np.mean(velocities)*dt
        contacts = List()
        empty = True
        for i in range(len(radius)):
            xi = positions[i]
            for j in range(i+1,len(radius)):
                xj = positions[j]
                distance = np.linalg.norm(xi-xj)-radius[i]-radius[j]
                if distance <= detection_range:
                    l = List()
                    l.append(i)
                    l.append(j)
                    contacts.append(l)
                    empty = False
            for j in range(len(walls)) :
                wall = walls[j].astype(np.float64)
                t = wall[1]-wall[0]
                s = xi-wall[0]
                st = np.dot(s,t)/np.linalg.norm(t)**2
                n = s-st*t
                d = np.linalg.norm(n)-radius[i]
                if d<detection_range and 0 <= st <= 1:
                    l = List()
                    l.append(i)
                    l.append(-(j+1))
                    contacts.append(l)
                    empty = False
                if 1 < st <= 1+radius[i]/np.linalg.norm(t) or -(radius[i])/np.linalg.norm(t) <= st < 0 :
                    print("eoh")
                    #contact with boundary disk
        if empty :
            l = List()
            l.append(-1)
            contacts.append(l)
        return contacts
    
    @staticmethod
    @jit(nopython=True)
    def solve_contacts(positions, velocities,omega, radius,imass,inertia, walls, contacts,restitution_wall ,dt):
        impulses = np.zeros(len(contacts))
        for k in range(len(contacts)) :
            i = int(contacts[k][0])
            j = int(contacts[k][1])
            xi = positions[i]
            if j < 0 : 
                #contact with wall
                wall = walls[-(j+1)].astype(np.float64)
                t = wall[1]-wall[0]
                s = xi-wall[0]
                st = np.dot(s,t)/np.linalg.norm(t)**2
                n = s-st*t
                velocities[i] = velocities[i] - (1+restitution_wall) * np.dot(velocities[i],n)/np.dot(n,n) * n
            
            else : 
                #contact with particle
                xj = positions[j]
                ui = velocities[i]
                uj = velocities[j]
                ri = radius[i]
                rj = radius[j]
                mi = 1/imass[i]
                mj = 1/imass[j]
                Ii = inertia[i]
                Ij = inertia[j]
                wi = omega[i]
                wj = omega[j]
                n = (xi-xj)/np.linalg.norm(xi-xj)
                t = np.array([n[1],-n[0],0])
                H = np.array([[n[0],n[1],0,-n[0],-n[1],0],
                              [t[0],t[1],ri,-t[0],-t[1],rj]])
                HT = np.transpose(H)
                Minverse = np.diag(np.array([1/mi,1/mi,1/Ii,1/mj,1/mj,1/Ij]))
                W = H@(Minverse@HT)
                W = np.diag(np.array([W[0,0],W[1,1]]))
                d = np.linalg.norm(xi-xj) - ri - rj
                vn = np.dot((ui-uj),n)
                vt = (wi-wj)
                dvn = max(0,-vn-max(0,d)/dt)
                mu = 0.3
                if vt*W[0,0] > mu*dvn*W[1,1]:
                    dvt = -mu*dvn*W[1,1]/W[0,0]
                else :
                    dvt = 0
                #dvt = 0
                dV = np.array([dvn,dvt])
                dv = Minverse@HT@np.linalg.inv(W)@dV

                M = mi+mj
                Pn = dvn/M
                velocities[i] = velocities[i] +  np.array([dv[0],dv[1],0])
                velocities[j] = velocities[j] +  np.array([dv[3],dv[4],0])
                impulses[k] = np.abs(Pn)      
        
        return impulses


    def step(self) -> None:
        """
        Update the positions and speeds of the particles using the velocity verlet algorithm
        """
        self.generate_tree(self.__positions)
        self.__velocities += self.g*self.dt
        converged = False
        self.contacts = self.detect_contacts(self.__positions,self.__velocities,self.__radius,self.lines,self.dt)
        while not converged : 
            if self.contacts[0][0] == -1 :
                converged = True
                break
            impulses = self.solve_contacts(self.__positions,self.__velocities,self.__omega,self.__radius,self.iM,
                                           self.I,self.lines,self.contacts,self.restitution_wall,self.dt)
            if np.sum(impulses) < 1e-3/(len(self.__radius)):
                converged = True
        self.__positions += self.__velocities * self.dt
        self.t += self.dt
        return