import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from matplotlib import collections  as mc

n_particle = 11*11
# le rayon du kernel
h = .5
xlim = 40

# Build a n_particle by 2 array containing the coordinates of all particles
X,Y = np.meshgrid(np.linspace(0, int(math.sqrt(n_particle)*h), int(math.sqrt(n_particle))),
                  np.linspace(0, int(math.sqrt(n_particle)*h), int(math.sqrt(n_particle))))
pos = np.array([X.flatten(), Y.flatten()]).T
# The array of x and y positions can be accessed with pos[:,0] and pos[:,1]

X1,Y1 = np.meshgrid(np.zeros(int(math.sqrt(n_particle))),
                    np.zeros(int(math.sqrt(n_particle))))
v = np.array([X1.flatten(), Y1.flatten()]).T
a = np.array([X1.flatten(), Y1.flatten()]).T

fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
maxX = ax.set_xlim(0, 40), ax.set_xticks([])
maxY = ax.set_ylim(0, 40), ax.set_yticks([])
scat = ax.scatter(pos[:, 1], pos[:, 0], s=0.000001, facecolors='blue')

lines = []
c = np.array([(1, 0, 0, 1)])
lc = mc.LineCollection(lines, colors=c, linewidths=2)
ax.add_collection(lc)

rho_ = np.zeros(n_particle)
vois_ = np.zeros(n_particle)
# la "densité au repos", pour le calcule de la pression (=rho_0)
d0 = 1000.
# la masse de chaque particule
m = d0*h*h
# la pesanteur
g = np.array([0., -9.81])
# le pas de temps
dt = 0.01
# le `nu` pour la viscosité
nu = 3

# juste un intermédiaire de calcul pour p0
c = 981
# p0: le `k` employé dans la formule de la pression
p0 = d0*c*c/7.

def calculKernel(d): # d : distance
    ''' Caclul de kernel, W(q) '''
    d = np.linalg.norm(d[0]-d[1])
    q = d / h 
    sigma = 10 / (7*np.pi*h**2)
    res = 0
    
    if q >= 0 and q <= 1 :
        
        res = sigma * (1 - 1.5*q**2 + 0.75*q**3)
    
    if q >= 1 and q <= 2 :
        
        res = sigma / 4 * (2 - q)**3
    
    return res


def calculKernelD(d) :
    ''' Calcul de 1/h W'(q) avec la dérivé de W(q) '''
    d = np.linalg.norm(d[0]-d[1])
    q = d / h 
    sigma = 10 / (7*np.pi*h**2)
    res = 0
    
    if q >= 0 and q <= 1 :
        
        res = sigma * (-3*q + 2.25*q**2)
    
    if q >= 1 and q <= 2 :
        
        res = -0.75 * sigma * (2 - q)**2
    
    
    return 1/h * res

def rho(i, grille) :
    res = 0
    p = pos[i]
    
    for j in proximite(i, grille) :
        
        kernel = calculKernel((p, pos[j]))
        res = res + ( m * abs(kernel) )
        
        
    return res


def calcul_gradient(d) :
    ''' calcul du gradient, prend en entrée une distance '''
    x = d[0]
    xj = d[1]
    
    vect_norm = np.linalg.norm(d[0] - d[1])
    if vect_norm < 0.000000001 :
        res = 0
    else :
        res = (x - xj) / vect_norm
    
    return res
    

def gradient_kernel(d):
    ''' calcul le gradient du kernel, prend en entée une distance '''
    deriv_ker = calculKernelD(d)
    grad = calcul_gradient(d)
    
    res = deriv_ker * grad
    return res
    

def calcul_pression(i):
    ''' calcul la pression d'une particule pi, prend en entrée une particule i'''
    global p0, d0


    res = p0*((rho_[i] / d0)**7 -1)
    if res < 0 : 
        res = 1
    
    return res


def force_pression(i, grille):
    ''' calcul de la force de pression pour une particule p(i)'''
    res = 0
    p = pos[i]
    
    for j in proximite(i, grille):
        
        
        gradient_kern = gradient_kernel((p, pos[j]))
        pi = calcul_pression(i)
        rho_i2 = (rho_[i]**2)
        pj = calcul_pression(j)
        rho_j2 = (rho_[i]**2)
        
        pi_pj_pj2 = ( (pi / rho_i2) + (pj / rho_j2) )
        
        res = res + ( (m*pi_pj_pj2) * gradient_kern )
        
    return res

def viscosite(i, grille) :
    ''' calcul la viscosité d'une particule p(i) '''
    res = 0 
    p = pos[i]
    vm = 2 * nu * m
    
    for j in proximite(i, grille):
        
        m_sur_rho = m / rho_[i]
        vit = v[j] - v[i]
        num = np.dot((pos[j] - p) , gradient_kernel((p, pos[j])))
        denum = np.linalg.norm(p - pos[j])**2 + (0.01*h**2)
        
        res = res + ( m_sur_rho * vit * (num / denum) )
        
    res = vm * res
    return res


def proximite(i, grille) :
    ''' renvoie une liste de voisin de la particule i '''
    voisin = []
    vois = []
    rayon = 2*h
    
    for j in range(len(pos)):
        
        pos_i = int(abs(pos[j, 0] / rayon))
        pos_j = int(abs(pos[j, 1] / rayon))
        
        particule = (pos_i, pos_j)  
        
        if i == j :
            
            voisin.append(grille[particule[0]-1][particule[1]-1])
            voisin.append(grille[particule[0]-1][particule[1]])
            if pos_j < 39 :
                voisin.append(grille[particule[0]-1][particule[1]+1])
            voisin.append(grille[particule[0]][particule[1]-1])
            voisin.append(grille[particule[0]][particule[1]])
            if pos_j < 39 :
                voisin.append(grille[particule[0]][particule[1]+1])
            if pos_i < 39 :
                voisin.append(grille[particule[0]+1][particule[1]-1])
            if pos_i < 39 :
                voisin.append(grille[particule[0]+1][particule[1]])
            if pos_j < 39 and pos_i < 39 :
                voisin.append(grille[particule[0]+1][particule[1]+1])


    while -1 in voisin :
        voisin.remove(-1)
 
    for k in range(len(voisin)):
        for n in range(len(voisin[k])):
            vois.append(voisin[k][n])
            
    return vois

def dist_signee(point):
    ''' calcul la particule la plus proche du point donné '''
    plus_proche = np.linalg.norm(point - pos[0])

    for j in range(len(pos)):
        if plus_proche > np.linalg.norm(point - pos[j])  :
            plus_proche = np.linalg.norm(point - pos[j])

    plus_proche -= 2*h

    return plus_proche


def grille_square():
    ''' créer une grille avec les positions à l'intérieur '''
    taille = xlim/(h*3)
    grille = []
    for i in range(int(taille)+1):
        
        for j in range(int(taille)+1):

            grille.append(np.array([i*(xlim/taille), j*(xlim/taille)]))
    return grille



def grille_dist(grille):
    ''' créer une grille en remplaçant la position par la distance avec le point le plus proche '''
    for i in range(len(grille)):
        dist = dist_signee(grille_square()[i])
        grille[i] = dist
    return grille


def square(grille):
    ''' regroupe les points de la grille en carré ( renvoie une liste contenant les 4 sommets de chaque carré )'''
    taille = int(xlim/(h*3))
    res = []

    for i in range(len(grille)-taille-2):
        carre = []

        carre.append(grille[i])
        carre.append(grille[i+1])
        carre.append(grille[i+(taille+2)])
        carre.append(grille[i+taille+1])
        

        res.append(carre)
    
    return res

def seg_changeSign(point1, point2):
    ''' vérifie si le segment est à la surface '''
    res = False
    if point1 < 0 and point2 > 0 or point1 > 0 and point2 < 0:
        res = True
    return res



def cube_changeSigne():
    ''' renvoie la liste de cube qui ont un chagement de signe '''

    grille = grille_square()
    spos = square(grille)
    dist = grille_dist(grille)
    sq = square(dist)
    res_dist = []
    res_pos= []

    for i in range(len(sq)):
        sqi = sq[i]

        for j in range(len(sqi)-1):
            dis = seg_changeSign(sqi[j], sqi[j+1])
            if dis and sqi not in res_dist or dis and sqi not in res_dist:
                if (sqi[j] + sqi[j+1] < 10) : 
                    res_dist.append(sqi)
                    res_pos.append(spos[i])
    
    return (res_dist, res_pos)


def interpolation(seg1, seg2):
    ''' calcul l'interpolation linéaire d'une arête '''
    return -seg1 /(seg2 - seg1)


def quatre_arretes(carre,i):
    ''' vérifie si le carré à 4 arretes '''
    carre = carre[0][i]
    res = False
    if carre[0] < 0 and carre[1] > 0 and carre[2] < 0 and carre[3] > 0 :
        res = True
    if carre[0] > 0 and carre[1] < 0 and carre[2] > 0 and carre[3] < 0 :
        res = True
    return res



def segment(carre,i):

    distance = carre[0][i]
    position = carre[1][i]

    seg1, seg2= 0, 0
    inter_lineaire_seg1, inter_lineaire_seg2= 0, 0

    line = []
    
    #test s'il y a 4 arêtes qui change de signe 
    if quatre_arretes(carre,i):
        #ajoute le premier élément à la fin pour avoir toutes les configurations
        distance.append(distance[0])
        position.append(position[0])
        
        seg3, seg4= 0, 0
        inter_lineaire_seg3, inter_lineaire_seg4= 0, 0

        
        for i in range(len(distance)-1):
            # fonction qui vérifie si on a un changement de signe sur cette arête
            if seg_changeSign(distance[i], distance[i+1]) :
                # créer le premier sommet
                if np.linalg.norm(seg1) == 0:
                    #calcul de l'interpolation linéaire des deux sommets
                    inter_lineaire_seg1 = interpolation(distance[i], distance[i+1])
                    #on multiplie les sommets l'un part t ( interpolation linéaire ) et l'autre par 1-t
                    #on les additionne pour avoir une moyenne 
                    seg1 = ( position[i] * (1-inter_lineaire_seg1)) + ( position[i+1] * inter_lineaire_seg1)

                #créer le deuxième sommet
                if np.linalg.norm(seg1) != 0: 
                    inter_lineaire_seg2 = interpolation(distance[i], distance[i+1])
                    seg2 = ( position[i] * (1-inter_lineaire_seg2)) + ( position[i+1] * inter_lineaire_seg2)

                #créer le troisième sommet
                if  np.linalg.norm(seg1) != 0 and np.linalg.norm(seg2) != 0: 
                    inter_lineaire_seg3 = interpolation(distance[i], distance[i+1])
                    seg3 = ( position[i] * (1-inter_lineaire_seg3)) + ( position[i+1] * inter_lineaire_seg3)
                
                #créer le dernier sommet
                if  np.linalg.norm(seg1) != 0 and np.linalg.norm(seg2) != 0 and np.linalg.norm(seg3) !=0: 
                    inter_lineaire_seg4 = interpolation(distance[i], distance[i+1])
                    seg4 = ( position[i] * (1-inter_lineaire_seg4)) + ( position[i+1] * inter_lineaire_seg4)

        line.append((seg1, seg2))
        line.append((seg3, seg4))
        return line
    
    else : 
        distance.append(distance[0])
        position.append(position[0])

        for i in range(len(distance)-1):
            if seg_changeSign(distance[i], distance[i+1]) :
                if np.linalg.norm(seg1) == 0:
                    inter_lineaire_seg1 = interpolation(distance[i], distance[i+1])
                    seg1 = ( position[i] * (1-inter_lineaire_seg1)) + ( position[i+1] * inter_lineaire_seg1)
                else : 
                    inter_lineaire_seg2 = interpolation(distance[i], distance[i+1])
                    seg2 = ( position[i] * (1-inter_lineaire_seg2)) + ( position[i+1] * inter_lineaire_seg2)

        line.append((seg1, seg2))
        return line


def trace_segment() :
    lines = []
    carre = cube_changeSigne()
    # on récupère la liste des carrés avec
    # au moins un changement de signe
    for i in range(len(carre[0])):
        # on calcul les sommets des segments à tracer 
        # pour tous les carrés
        for j in segment(carre,i):
            # on ajoute les segments dans la liste
            lines.append(j)

    return lc.set_segments(lines)



    

def animate(i):
    # Makes the points advance in x position, for demonstration purpose
    global dt, m, g
    rayon = 2*h
    
    
    #créer une grille 
    grille = int(xlim/rayon) * [-1]
    for l in range(len(grille)):
        grille[l] = int(xlim/rayon) * [-1]
    
    #remplis la grille avec l'indice des particules
    for k in range(len(pos)):
        pos_i = int(abs(pos[k, 0] / rayon))
        pos_j = int(abs(pos[k, 1] / rayon))
        
        #s'il n'y a pas de particule, alors on créer un tableau de 1 seule particule
        if grille[pos_i][pos_j] == -1 :
            grille[pos_i][pos_j] = [k]
            
        #sinon on ajoute la particule au tableau déjà crée 
        else :
            grille[pos_i][pos_j].append(k)
    
    
    for i in range(len(pos)):
        
        proximite(i, grille)
        
        rho_[i] = rho(i, grille)
        
        force_pres = -(force_pression(i, grille) / m)
        viscositee = (viscosite(i, grille) / m)
        v[i] = (viscositee + force_pres+ g) * dt + v[i]
        pos[i] = v[i] * dt + pos[i]
    

        if pos[i,0] >= maxX[0][1] or pos[i,0] <= maxX[0][0] :
            v[i,0] = -v[i,0]
            pos[i,0] = v[i,0] * dt + pos[i,0]
        
    
        if pos[i,1] >= maxY[0][1] or pos[i,1] <= maxY[0][0] :
            v[i, 1] = -v[i, 1]
            pos[i, 1] = v[i, 1] * dt + pos[i, 1]
    if i % 5 == 0 :
        trace_segment()
    # Update the plot with the current positions
    scat.set_offsets(pos)
    scat.set_array(rho_)

# Define and run the animation.
# Anything we want to change from there should be called from the animate
# function
ani = animation.FuncAnimation(fig, animate, interval=0)
plt.show()