import numpy as np
import numba
from numba import deferred_type, optional, float64, int64,typeof
from numba.experimental import jitclass
from numba.typed import List

node_type = deferred_type()
spec = (
    ("npoints", int64),
    ("points", float64[:,:] ),
    ("center", float64[:]),
    ("length", float64),
    ("nw", optional(node_type)),
    ("ne", optional(node_type)),
    ("sw", optional(node_type)),
    ("se", optional(node_type)),
    ("max",int64),
    ("IDs",int64[:]),
    ("leaf",int64),
)

@jitclass(spec)
class QuadTreeNode(object):

    def __init__(self, center, length):
        self.max = 4   #max number of points in the node
        self.points = np.zeros((self.max,3))   #on note pas les sous arrays sinon numba plante
        self.IDs = np.zeros(self.max).astype(int64)
        self.npoints = 0
        self.center = center       #center of the node
        self.length = length       #length of the node
        self.leaf= 1

        self.nw = None             #top left node
        self.ne = None             #top right node
        self.sw = None             #bottom left node
        self.se = None             #bottom right node

node_type.define(QuadTreeNode.class_type.instance_type)

@numba.jit() #ok
def add_point(node, point,i):
    stack_node = List()
    stack_point = List()
    stack_i = List()
    stack_node.append(node)
    stack_point.append(point)
    stack_i.append(i)

    while len(stack_node):
        node = stack_node[0]
        point = stack_point[0]
        i = stack_i[0]
        stack_node.pop(0)
        stack_point.pop(0)
        stack_i.pop(0)
        xmin = node.center[0]-node.length/2
        xmax = node.center[0]+node.length/2
        ymin = node.center[1]-node.length/2
        ymax = node.center[1]+node.length/2

        if point[0] <= xmin or point[0]>xmax or point[1] <= ymin or point[1]>ymax:                #Check if point is outside node
            continue

        if node.leaf == 0 :
            stack_node.append(node.nw)
            stack_node.append(node.sw)
            stack_node.append(node.se)
            stack_node.append(node.ne)

            stack_i.append(i)
            stack_i.append(i)
            stack_i.append(i)
            stack_i.append(i)

            stack_point.append(point)
            stack_point.append(point)
            stack_point.append(point)
            stack_point.append(point)

        elif node.npoints < node.max:
            node.points[node.npoints] = point
            node.IDs[node.npoints] = i
            node.npoints += 1
        else :
            set_nodes(node)
            points_toremove,IDs = get_points(node)
            node.leaf = 0
            stack_node.append(node)
            stack_i.append(i)
            stack_point.append(point)
            for j in range(len(points_toremove)) :
                p = points_toremove[j]
                ID = IDs[j]
                stack_node.append(node)
                stack_i.append(ID)
                stack_point.append(stack_point[-1])
    return

@numba.jit(nopython = True) #ok
def set_nodes(node):
    l = node.length / 2
    d = node.length / 4

    node.nw = QuadTreeNode(node.center + np.array((-d, d)), l)
    node.ne = QuadTreeNode(node.center + np.array((d, d)), l)
    node.sw = QuadTreeNode(node.center + np.array((-d, -d)), l)
    node.se = QuadTreeNode(node.center + np.array((d, -d)), l)

    return l

@numba.jit(nopython = True) #ok
def get_points(node):
    points = node.points[:node.npoints]
    IDs = node.IDs[:node.npoints]
    return points,IDs

@numba.jit(nopython = True) #ok
def point_in_square(point,center,length):
    #Check if a point is in a square
    x = point[0]
    y = point[1]
    max_x = center[0]+length/2
    min_x = center[0]-length/2
    max_y = center[1]+length/2
    min_y = center[1]-length/2
    if min_x <= x <= max_x and min_y <= y <= max_y:
        return True
    else :
        return False

@numba.jit(nopython = True) #ok
def intersects(centerA,lengthA,centerB,lengthB):
    #Return true if rectangle A intersects rectangle B
    xminA = centerA[0] - lengthA/2
    xmaxA = centerA[0] + lengthA/2
    yminA = centerA[1] - lengthA/2
    ymaxA = centerA[1] + lengthA/2

    xminB = centerB[0] - lengthB/2
    xmaxB = centerB[0] + lengthB/2
    yminB = centerB[1] - lengthB/2
    ymaxB = centerB[1] + lengthB/2

    return not (xmaxA<xminB or xmaxB<xminA or ymaxA < yminB or ymaxB<yminA)

@numba.jit()
def nodes_in_square(node,point,L,lst):
    #return all the leaf nodes that touches the square of side L centered on point
    if intersects(node.center,node.length,point,L):
        if node.leaf == 1:
            lst.append(node)
        else :
            nodes_in_square(node.nw,point,L,lst)
            nodes_in_square(node.sw,point,L,lst)
            nodes_in_square(node.ne,point,L,lst)
            nodes_in_square(node.se,point,L,lst)

@numba.jit()
def particles_in_box(node,point,L):
    #return all particles coordinates in the box of size L around point
    nodes = List()
    nodes.append(QuadTreeNode(np.array([0.0,0.0,0.0]),0.0))
    nodes_in_square(node,point,L,nodes)
    nparticles = 0
    particles = np.zeros((node.max*len(nodes),3))
    IDs = np.zeros(node.max*len(nodes))
    for i in range(1,len(nodes)):
        points,ids = get_points(nodes[i])
        particles[nparticles:nparticles+nodes[i].npoints,] = points
        IDs[nparticles:nparticles+nodes[i].npoints] = ids
        nparticles += nodes[i].npoints
    return particles[:nparticles],IDs[:nparticles]

@numba.jit()
def set_tree(points):
    xmin = np.min(points[:,0]) - 1
    xmax = np.max(points[:,0]) + 1
    ymin = np.min(points[:,1]) - 1
    ymax = np.max(points[:,1]) + 1
    center = np.array([xmax+xmin,ymax+ymin])/2
    length = np.max(np.array([xmax-xmin,ymax-ymin]))
    tree = QuadTreeNode(center,length)
    for i in range(len(points)):
        add_point(tree,points[i],i)
    return tree
