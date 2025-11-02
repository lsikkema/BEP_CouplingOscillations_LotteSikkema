# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 18:13:07 2025

@author: Lotte
"""

import matplotlib.pyplot as plt
import numpy as np

# define constants
ag = 10
k= 1
n = 10

a3g = 10
coupling_strength = 0.7

period = 0.1
w =  2*np.pi/period
t_final = 100
dt = 0.1
t = np.arange(0,t_final,dt)
N_cells = 44

# model
def dd_dt(Hp,D):
    x = (ag * k**n)/(k**n + Hp**n) #Hes protein inhibition
    x -= D
    return x

def dhm_dt(D,Hm,Hp, t): 
    D2 = np.roll(D, 1)
    D0 = np.roll(D,-1)
    
    y = ag*coupling_strength*(D2+D0)/2 #linear transactivation
    y += (1-coupling_strength)*((ag * k**n)/(k**n + D**n)) #cisinhibition
    y -= Hm #degradation
    return y

def dhp_dt(Hm,Hp,t): 
    z = a3g*Hm #linear Hes mRNA activation
    z *= np.sin(w*t) #forcing Hes stimulation into sine
    z -= Hp #degredation
    return z

#simulate
def Euler(x, dx_dt, dt):
    x_new = x + dx_dt*dt    
    return x_new

def sim_lattice(d0, hm0, hp0):
    D = np.zeros((len(t),N_cells))
    Hm = np.zeros((len(t),N_cells))
    Hp = np.zeros((len(t),N_cells))
        
    D[0,:] = d0
    Hm[0,:] = hm0
    Hp[0,:] = hp0
    
    for i, ti in enumerate(t[:-1]):
        dd = dd_dt(hp0,d0)
        dhm = dhm_dt(d0,hm0,hp0, ti*dt) #hp0) # ti*dt)
        dhp = dhp_dt(hm0,hp0, ti*dt) 
        
        d0 = Euler(d0,dd,dt)
        hm0 = Euler(hm0,dhm,dt)
        hp0 = Euler(hp0,dhp,dt)   
       
        d0[0] = d0[1] = d0[-1] = d0[-2] = 0 #onthou dat er dus 4 extra cellen zijn
        hm0[0] = hm0[1] = hm0[-1] = hm0[-2] = 0
        hp0[0] = hp0[1] = hp0[-1] = hp0[-2] = 0
        
        D[i,:] = d0
        Hm[i,:] = hm0
        Hp[i,:] = hp0

    
    return D,Hm,Hp

# initial values
hm0 = np.random.rand(N_cells)
d0 = np.random.rand(N_cells)
hp0 = np.random.rand(N_cells)


D,Hm,Hp = sim_lattice(d0,hm0,hp0)

# plot

plt.figure(figsize=(10,6))
plt.plot(t, Hm, label="Hes mRNA", color='gold', alpha=0.5)
plt.plot(t, Hp, label="Hes protein", color='crimson', alpha = 0.5)
plt.plot(t, D, label="Delta", color='royalblue', alpha = 0.5)
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title(f"Temporal oscillations, period={period}, coupling_strength={coupling_strength}")

plt.figure(figsize=(10,6))
plt.pcolor(D)
plt.title(f'Spatial oscillation Delta, Hes period={period}, coupling_strength={coupling_strength}')
plt.xlabel("Grid cells")
plt.ylabel("Time")

'''
plt.figure(figsize=(10,6))
plt.plot(t, D[:,2], label="Delta, cell 1", color='royalblue', alpha = 0.5)
plt.plot(t, Hm[:,2], label="Hes mRNA, cell 1", color='gold')
plt.plot(t, Hp[:,2], label="Hes protein, cell 1", color='crimson', alpha = 0.5)
plt.plot(t, D[:,3], label="Delta, cell 2", color='cornflowerblue', alpha = 0.5)
plt.plot(t, Hm[:,3], label="Hes mRNA, cell 2", color='goldenrod')
plt.plot(t, Hp[:,3], label="Hes protein, cell 2", color='indianred', alpha = 0.5)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Temporal oscillations") #" #$\omega$ = {w}")

plt.figure(figsize=(10,6))
plt.pcolor(Hp)
plt.title('Spatial oscillations, Hes protein')
plt.xlabel("Grid cells")
plt.ylabel("Time")

plt.figure(figsize=(10,6))
plt.pcolor(Hm)
plt.title('Spatial oscillations, Hes mRNA')
plt.xlabel("Grid cells")
plt.ylabel("Time")

plt.figure(figsize=(10,6))
plt.pcolor(D)
plt.title('Spatial oscillations, Delta')
plt.xlabel("Grid cells")
plt.ylabel("Time")

plt.figure(figsize=(10,6))
plt.plot(t, Hp, label="Hes protein", color='crimson', alpha = 0.5)
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Hes protein concentrations over time")

plt.figure(figsize=(10,6))
plt.plot(t, Hm, label="Hes mRNA", color='gold', alpha = 0.5)
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Hes mRNA concentrations over time")

plt.figure(figsize=(10,6))
plt.plot(t, D, label="Delta", color='royalblue', alpha = 0.5)
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Delta concentrations over time")
'''

def phaseplane(D,N,H, f,g,k, x,y,z):
    d_vals = np.linspace(0, x, x)
    n_vals = np.linspace(0, y, y)
    h_vals = np.linspace(0, z, z)
    
    Dg, Ng, Hg = np.meshgrid(d_vals, n_vals, h_vals)
    
    dD = f(Hg, Dg)
    dN = g(Dg, Ng)
    dH = k(Ng, Hg)
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(Dg, Ng, Hg, dD, dN, dH, length=1, normalize=True, color='grey', alpha=0.8)
    ax.plot(D, N, H, color='red') 

    ax.plot(D, N, H, color='red')
    ax.set_xlabel("D")
    ax.set_xlim(0,x)
    ax.set_ylabel("N")
    ax.set_ylim(0,y)
    ax.set_zlabel("H")
    ax.set_zlim(0,z)
    ax.set_title("3D Phase Space Trajectory (D-N-H)")

    return plt.show()





