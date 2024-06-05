from numba import jit, int64, float64
from numba.experimental import jitclass
import numpy as np


CONTACT_PARTICLE_PARTICLE = 1
CONTACT_PARTICLE_LINE = 2
CONTACT_PARTICLE_RECTANGLE = 3

@jitclass([("i", int64), ("j", int64), ("type", int64), ("normal", float64[:])])
class Contact:
    def __init__(self, i, j, normal, type=CONTACT_PARTICLE_PARTICLE):
        self.i = i
        self.j = j
        self.normal = normal
        self.type = type

@jit(nopython=True)
def solve_contacts(positions, velocities, omega, radius, imass, inertia, contacts, restitution_wall, dt):
    impulses = np.zeros(len(contacts))
    for k in range(len(contacts)) :
        contact = contacts[k]
        i, j = contact.i, contact.j
        xi = positions[i]
        if contact.type == CONTACT_PARTICLE_LINE:
            velocities[i] = velocities[i] - (1 + restitution_wall) * np.dot(velocities[i], contact.normal) / np.dot(contact.normal, contact.normal) * contact.normal
        elif contact.type == CONTACT_PARTICLE_RECTANGLE:
            velocities[i] = velocities[i] - (1 + restitution_wall) * np.dot(velocities[i], contact.normal) * contact.normal
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
            n = contact.normal
            t = np.array([n[1], -n[0], 0])
            H = np.array([[n[0], n[1], 0, -n[0], -n[1], 0],
                            [t[0], t[1], ri, -t[0], -t[1], rj]])
            HT = np.transpose(H)
            Minverse = np.diag(np.array([1 / mi, 1 / mi, 1 / Ii, 1 / mj, 1 /mj, 1 / Ij]))
            W = H @ (Minverse @ HT)
            W = np.diag(np.array([W[0,0], W[1,1]]))
            d = np.linalg.norm(xi - xj) - ri - rj
            vn = np.dot((ui - uj), n)
            vt = (wi - wj)
            dvn = max(0, -vn - max(0, d) / dt)
            mu = 0.3
            if vt * W[0,0] > mu * dvn *W[1,1]:
                dvt = -mu * dvn * W[1,1] / W[0,0]
            else:
                dvt = 0
            #dvt = 0
            dV = np.array([dvn,dvt])
            dv = Minverse @ HT @ np.linalg.inv(W) @ dV

            M = mi+mj
            Pn = dvn/M
            velocities[i] = velocities[i] +  np.array([dv[0], dv[1], 0])
            velocities[j] = velocities[j] +  np.array([dv[3], dv[4], 0])
            impulses[k] = np.abs(Pn)

    return impulses

@jit(nopython=True)
def detect_contacts(positions, velocities, radius, lines, rectangles, dt):
    detection_range = np.mean(velocities) * dt
    contacts = []
    for i in range(len(radius)):
        detect_contact_particles(i, positions, radius, detection_range, contacts)
        detect_contact_lines(i, positions[i], radius[i], lines, detection_range, contacts)
        detect_contact_rectangle(i, positions[i], radius[i], rectangles, detection_range, contacts)
    if not contacts:
        contacts.append(Contact(-1, -1, np.zeros(3), -1))

    return contacts

@jit(nopython=True)
def detect_contact_particles(i, positions, radius, detection_range, contacts):
    xi = positions[i]
    for j in range(i + 1, len(radius)):
        xj = positions[j]
        distance = np.linalg.norm(xi - xj) - radius[i] - radius[j]
        if distance <= detection_range:
            contacts.append(Contact(i, j, (xi- xj) / np.linalg.norm(xi - xj), CONTACT_PARTICLE_PARTICLE))

@jit(nopython=True)
def detect_contact_lines(i, xi, rad, lines, detection_range, contacts):
    for j in range(len(lines)) :
        line = lines[j].astype(np.float64)
        t = line[1] - line[0]
        if np.linalg.norm(t) < 1e-9:
            continue

        s = xi - line[0]
        st = np.dot(s, t) / np.linalg.norm(t) ** 2
        n = s - st * t
        d = np.linalg.norm(n) - rad
        if d < detection_range and 0 <= st <= 1:
            contact = Contact(i, j, n, CONTACT_PARTICLE_LINE)
            contacts.append(contact)

        if 1 < st <= 1 + rad / np.linalg.norm(t) or - rad / np.linalg.norm(t) <= st < 0 :
                print("eoh")
                #contact with boundary disk

@jit(nopython=True)
def detect_contact_rectangle(i, xi, rad, rectangles, detection_range, contacts):
    for k in range(len(rectangles)):
        rectangle = rectangles[k]
        ab, ac = rectangle[1] - rectangle[0], rectangle[2] - rectangle[0]
        n = np.cross(ab, ac)
        if np.linalg.norm(n) < 1e-6:
            continue

        n = n / np.linalg.norm(n)
        d = -np.dot(n, rectangle[0])
        dst_plane_dir = (n[0] * xi[0] + n[1] * xi[1] + n[2] * xi[2] + d)
        dst_plane = np.abs(dst_plane_dir)
        h = xi - dst_plane_dir * n
        dst_h_rect = np.inf
        for j in range(4):
            b, c = rectangle[j], rectangle[(j+1)%4]
            vecd = (c - b) / np.linalg.norm(c - b)
            v = h - b
            t = np.dot(v, vecd)
            p = b + t * vecd
            dist = np.linalg.norm(p - h)
            if dist < dst_h_rect:
                dst_h_rect = dist

        dst = dst_plane + (dst_h_rect if not inside_rectangle(rectangle, h) else 0)
        if dst - rad < detection_range:
            normal = (xi - h) / np.linalg.norm(xi - h)
            contacts.append(Contact(i, k, normal, CONTACT_PARTICLE_RECTANGLE))

@jit(nopython=True)
def inside_rectangle(rectangle, point):
    mn_x = min(rectangle[0][0], rectangle[1][0], rectangle[2][0], rectangle[3][0])
    mx_x = max(rectangle[0][0], rectangle[1][0], rectangle[2][0], rectangle[3][0])
    mn_y = min(rectangle[0][1], rectangle[1][1], rectangle[2][1], rectangle[3][1])
    mx_y = max(rectangle[0][1], rectangle[1][1], rectangle[2][1], rectangle[3][1])
    mn_z = min(rectangle[0][2], rectangle[1][2], rectangle[2][2], rectangle[3][2])
    mx_z = max(rectangle[0][2], rectangle[1][2], rectangle[2][2], rectangle[3][2])
    return mn_y <= point[1] <= mx_y and mn_z <= point[2] <= mx_z and mn_x <= point[0] <= mx_x