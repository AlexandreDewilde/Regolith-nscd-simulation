import numpy as np
from .tree import *
import quads

CONTACT_PARTICLE_PARTICLE = 0
CONTACT_PARTICLE_LINE = 1
CONTACT_PARTICLE_DISK = 2

class Contact:
    def __init__(self, i, j, normal,d, type=CONTACT_PARTICLE_PARTICLE):
        self.i = i
        self.j = j
        self.normal = normal
        self.d = d
        self.type = type

def norm_with_axis1(vector):
    array = np.zeros(vector.shape[0])
    for i in range(len(array)):
        array[i] = np.linalg.norm(vector[i])
    return array

def sum_with_axis1(vector):
    array = np.zeros(vector.shape[0])
    for i in range(len(array)):
        array[i] = np.sum(vector[i])
    return array

def solve_contacts_jacobi(contacts,positions, velocities, omega, radius, imass, inertia, dt):
    mu = 0.3
    m = 1/imass
    itmax = 2
    tol = 5e-6
    if contacts[0].type == -1 :
        return velocities,omega
    it = 0

    dvsn = np.ones((len(contacts),3))*2*tol/dt                                  #Array containing the normal velocity corrections times the normals
    dvst = np.ones((len(contacts),3))*2*tol/dt                                  #Array containing the tangential velocity corrections times the tangents
    dvsp = np.zeros((len(contacts),1))                                          #Array containing the tangential velocity corrections
    types = np.array([c.type for c in contacts])                                   #Array containing the type of each contact
    ncg = types[types==0].shape[0]                                                #Array containing the number of grain-grain contacts
    ncs = types[types==1].shape[0]                                                #Array containing the number of grain-segment contacts
    ncd = types[types==2].shape[0]                                                #Array containing the number of grain-disk contacts
    i0 = np.array([c.i for c in contacts])                                     #Array containing the indices of the first object implied in each contact
    i1 = np.array([c.j for c in contacts])                                     #Array containing the indices of the second object implied in each contact
    wnn = imass[i0]                                                            #Computing wnn, here only ok for grain-segment and grain-disk contacts
    wnn[types==0] = (m[i0[types==0]] + m[i1[types==0]])/(m[i0[types==0]]*m[i1[types==0]]) #Modifying wnn to the correct value for grain-grain contacts
    a = inertia[0]/(m[0]*radius[0]**2)                                    #Computing the moment of inertia factor which is the same for eavery grain
    wtt = (1+a)/(a*m[i0])                                                    #Computing wtt, here only ok for grain-segment and grain-disk contacts
    wtt[types==0] = a*(m[i0[types==0]] + m[i1[types==0]])/(m[i0[types==0]]*m[i1[types==0]]) #Modifying wtt to the correct value for grain-grain contacts
    
 
    if ncg > 0:                                                                   #Extracting grain-grain contact data
        ng = np.zeros((ncg,3))
        dg = np.zeros(ncg)
        tg = np.zeros((ncg,3))                                                   
        ncgi = 0
    
    if ncs > 0:                                                             #Extracting grain-segment contact data
        ns = np.zeros((ncs,3))
        ds = np.zeros(ncs)
        ts = np.zeros((ncs,3))
        ncsi = 0

    if ncd > 0:                                                             #Extracting grain-disk contact data
        nd = np.zeros((ncd,3))
        dd = np.zeros(ncd)
        td = np.zeros((ncd,3))
        ncdi = 0


    for i in range(len(contacts)):   
        c = contacts[i]
        if types[i] == 0:
            ng[ncgi] = c.normal                                              #Array containing the normal for each grain-grain contact
            dg[ncgi] = c.d                                                   #Array containing the distances for each grain-grain contact
            tg[ncgi] = np.array([c.normal[1],-c.normal[0],0])                #Array containing the tangent for each grain-grain contact
            ncgi += 1
        
        if types[i] == 1:
            ns[ncsi] = c.normal                                              #Array containing the normal for each grain-segment contact
            ds[ncsi] = c.d                                                   #Array containing the distances for each grain-segment contact
            ts[ncsi] = np.array([c.normal[1],-c.normal[0],0])                #Array containing the tangent for each grain-segment contact
            ncsi += 1

        if types[i] == 2:
            nd[ncdi] = c.normal                                              #Array containing the normal for each grain-segment contact
            dd[ncdi] = c.d                                                   #Array containing the tangent for each grain-segment contact
            td[ncdi] = np.array([c.normal[1],-c.normal[0],0])                #Array containing the distances for each grain-segment contact            
            ncdi += 1


    while np.any((norm_with_axis1(dvsn)**2 + norm_with_axis1(dvst)**2)**0.5*dt > tol) and it < itmax:             #Check for convergence AND if the number of iteration is below the maximum
      it += 1
      if ncg > 0:                                                                                                 #If there are grain-grain contacts
        vrelg = velocities[i0[types==0],:] - velocities[i1[types==0],:]                                                   #Compute the relative velocities for each grain-grain contact
        vng = np.sum(vrelg*ng,axis=1)                                                                             #Compute the normal component of the relative velocity for each grain-grain contact
        vtg = np.sum(vrelg*tg,axis=1) - omega[i0[types==0]]*radius[i0[types==0]] - omega[i1[types==0]]*radius[i1[types==0]] #Compute the tangential component of the relative velocity for each grain-grain contact
        dvgn = np.maximum(0,-vng - np.maximum(dg,0)/dt)                                                           #Compute the normal velocity correction for each grain-grain contact
        dvgt = np.where(np.abs(vtg)*wnn[types==0] <= mu*dvgn*wtt[types==0],-vtg,-mu*dvgn*vtg*(wtt[types==0]/wnn[types==0])/np.abs(vtg)) #Compute the tangential velocity correction for each grain-grain contact

        dvst[types==0,:] = np.expand_dims(dvgt[:],-1)*tg                                                                             #Store the tangential velocity correction times the tangent vector for each grain-grain contact
        dvsn[types==0,:] = np.expand_dims(dvgn[:],-1)*ng                                                                             #Store the normal velocity correction times the normal vector for each grain-grain contact
        dvsp[types==0,0] = dvgt                                                                                   #Store the tangential velocity correction for each grain-grain contact

      if ncs > 0:                                                                                                 #If there are grain-segment contacts
        vrels = velocities[i0[types==1],:]                                                                       #Compute the relative velocities for each grain-segment contact
        vns = np.sum(vrels*ns,axis=1)                                                                             #Compute the normal component of the relative velocity for each grain-segment contact
        vts = np.sum(vrels*ts,axis=1) - omega[i0[types==1]]*radius[i0[types==1]]                             #Compute the tangential component of the relative velocity for each grain-segment contact
        dvssn = np.maximum(0,-vns - np.maximum(ds,0)/dt)                                                          #Compute the normal velocity correction for each grain-segment contact
        dvsst = np.where(np.abs(vts)*wnn[types==1] <= mu*dvssn*wtt[types==1],-vts,-mu*dvssn*vts*(wtt[types==1]/wnn[types==1])/np.abs(vts)) #Compute the tangential velocity correction for each grain-segment contact
        dvst[types==1,:] = np.expand_dims(dvsst[:],-1)*ts                                                                       #Store the tangential velocity correction times the tangent vector for each grain-segment contact
        dvsn[types==1,:] = np.expand_dims(dvssn[:],-1)*ns                                                                       #Store the normal velocity correction times the normal vector for each grain-segment contact
        dvsp[types==1,0] = dvsst                                                                                  #Store the tangential velocity correction for each grain-segment contact

      if ncd > 0:                                                                                                 #If there are grain-disk contacts
        vreld = velocities[i0[types==2],:]                                                                       #Compute the relative velocities for each grain-disk contact
        vnd = np.sum(vreld*nd,axis=1)                                                                             #Compute the normal component of the relative velocity for each grain-disk contact
        vtd = np.sum(vreld*td,axis=1) - omega[i0[types==2]]*radius[i0[types==2]]                             #Compute the tangential component of the relative velocity for each grain-disk contact
        dvdn = np.maximum(0,-vnd - np.maximum(dd,0)/dt)                                                           #Compute the normal velocity correction for each grain-disk contact
        dvdt = np.where(np.abs(vtd)*wnn[types==2] <= mu*dvdn*wtt[types==2],-vtd,-mu*dvdn*vtd*(wtt[types==2]/wnn[types==2])/np.abs(vtd)) #Compute the tangential velocity correction for each grain-disk contact
        dvst[types==2,:] = np.expand_dims(dvdt[:],-1)*td                                                                        #Store the tangential velocity correction times the tangent vector for each grain-disk contact
        dvsn[types==2,:] = np.expand_dims(dvdn[:],-1)*nd                                                                        #Store the normal velocity correction times the normal vector for each grain-disk contact
        dvsp[types==2,0] = dvdt                                                                                   #Store the tangential velocity correction for each grain-disk contact

      pcx0n = np.bincount(i0,weights=(dvsn/np.expand_dims(wnn,-1))[:,0],minlength=velocities.shape[0])                                   #For each grain compute the x impulse for all the contacts in which it was grain 0 from normal corrections
      pcy0n = np.bincount(i0,weights=(dvsn/np.expand_dims(wnn,-1))[:,1],minlength=velocities.shape[0])                                   #For each grain compute the y impulse for all the contacts in which it was grain 0 from normal corrections
      pcx1n = np.bincount(i1[types==0],weights=(dvsn[types==0]/np.expand_dims(wnn[types==0],-1))[:,0],minlength=velocities.shape[0])     #For each grain compute the x impulse for all the contacts in which it was grain 1 from normal corrections
      pcy1n = np.bincount(i1[types==0],weights=(dvsn[types==0]/np.expand_dims(wnn[types==0],-1))[:,1],minlength=velocities.shape[0])     #For each grain compute the y impulse for all the contacts in which it was grain 1 from normal corrections
      pcx0t = np.bincount(i0,weights=(dvst/np.expand_dims(wtt,-1))[:,0],minlength=velocities.shape[0])                                   #For each grain compute the x impulse for all the contacts in which it was grain 0 from tangential corrections
      pcy0t = np.bincount(i0,weights=(dvst/np.expand_dims(wtt,-1))[:,1],minlength=velocities.shape[0])                                   #For each grain compute the y impulse for all the contacts in which it was grain 0 from tangential corrections
      pcx1t = np.bincount(i1[types==0],weights=(dvst[types==0]/np.expand_dims(wtt[types==0],-1))[:,0],minlength=velocities.shape[0])     #For each grain compute the x impulse for all the contacts in which it was grain 1 from tangential corrections
      pcy1t = np.bincount(i1[types==0],weights=(dvst[types==0]/np.expand_dims(wtt[types==0],-1))[:,1],minlength=velocities.shape[0])     #For each grain compute the y impulse for all the contacts in which it was grain 1 from tangential corrections
      ptc0 = np.bincount(i0,weights=(dvsp/np.expand_dims(wtt,-1))[:,0],minlength=omega.shape[0])                                    #For each grain compute the angular impulse for all the contacts in which it was grain 0
      ptc1 = np.bincount(i1[types==0],weights=(dvsp[types==0]/np.expand_dims(wtt[types==0],-1))[:,0],minlength=omega.shape[0])      #For each grain compute the angular impulse for all the contacts in which it was grain 1
      velocities[:,0] += (pcx0n + pcx0t - pcx1n - pcx1t)/m[:]                                                  #Update the x velocity of each grain
      velocities[:,1] += (pcy0n + pcy0t - pcy1n - pcy1t)/m[:]                                                  #Update the y velocity of each grain
      omega[:] -= radius[:]*(ptc0 + ptc1)/inertia[:]                                                        #Update the angular velocity of each grain"""

    return velocities,omega

def detect_contacts(positions, velocities, radius, walls, dt,tree):
    detection_range = np.max(radius)
    contacts = []
    empty = True
    if tree == None: 
        for i in range(len(radius)):
            xi = positions[i]
            for j in range(i+1,len(radius)):
                xj = positions[j]
                distance = np.linalg.norm(xi-xj)-radius[i]-radius[j]
                if distance <= detection_range:
                    c = Contact(i,j,(xi-xj)/np.linalg.norm(xi-xj),distance,0)
                    contacts.append(c)
                    empty = False
            for j in range(len(walls)) :
                wall = walls[j].astype(np.float64)
                t = wall[1]-wall[0]
                s = xi-wall[0]
                st = np.dot(s,t)/np.linalg.norm(t)**2
                n = s-st*t
                d = np.linalg.norm(n)-radius[i]
                if d<detection_range and 0 <= st <= 1:
                    c = Contact(i,j,n/(np.linalg.norm(n)),d,1)
                    contacts.append(c)
                    empty = False
                disk0 = wall[0]
                disk1 = wall[1]
                distance0 = np.linalg.norm(xi - disk0) - radius[i]
                distance1 = np.linalg.norm(xi - disk1) - radius[i]
                if distance0 < detection_range :
                    contacts.append(Contact(i,j,n,d,2))
                if distance1 < detection_range :
                    contacts.append(Contact(i,j,n,d,2))

    else :
        for i in range(len(radius)):
            xi = positions[i]
            particles,IDs = particles_in_box(tree,xi,detection_range*2)
            for j in range(len(IDs)):
                if int(IDs[j]) <= i:
                    continue
                xj = positions[int(IDs[j])]
                distance = np.linalg.norm(xi-xj)-radius[i]-radius[int(IDs[j])]
                if distance <= detection_range:
                    c = Contact(i,int(IDs[j]),(xi-xj)/np.linalg.norm(xi-xj),distance,0)
                    contacts.append(c)
                    empty = False
            for j in range(len(walls)) :
                wall = walls[j].astype(np.float64)
                t = wall[1]-wall[0]
                s = xi-wall[0]
                st = np.dot(s,t)/np.linalg.norm(t)**2
                n = s-st*t
                d = np.linalg.norm(n)-radius[i]
                if d<detection_range and 0 <= st <= 1:
                    c = Contact(i,j,n/(np.linalg.norm(n)),d,1)
                    contacts.append(c)
                    empty = False
                disk0 = wall[0]
                disk1 = wall[1]
                distance0 = np.linalg.norm(xi - disk0) - radius[i]
                distance1 = np.linalg.norm(xi - disk1) - radius[i]
                if distance0 < detection_range :
                    contacts.append(Contact(i,j,n,d,2))
                if distance1 < detection_range :
                    contacts.append(Contact(i,j,n,d,2))
                    
    if empty :
        c = Contact(-1,-1,np.array([-1.0,-1.0,-1.0]),-1.0,-1)
        contacts.append(c)
    print(len(contacts))
    return contacts 
