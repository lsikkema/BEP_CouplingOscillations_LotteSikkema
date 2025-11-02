# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 21:08:55 2025

@author: Lotte
"""

import numpy as np
import matplotlib.pyplot as plt

# define constants
a1, a2, a3 = 1, 1, 1
g1, g2 = 0.5, 0.5
k1, k2 = 1, 1
n, m = 10, 10

t_final = 100
dt = 0.1
t = np.arange(0, t_final, dt)

coupling_strength = 0

period = 1
w = 2 * np.pi / period                  
N_cells = 44             
x = np.linspace(0, 2*np.pi, N_cells)   
k = 4  

# model
def dd_dt(H, D):
    return a1 * (k1**n / (H**n + k1**n)) - g1 * D

def dh_dt(D, H): # phi):
    D0 = np.roll(D, 1)
    D2 = np.roll(D, -1)
    
    # nonlinear transactivation
    z = coupling_strength * a2 * (((D0 + D2)/2)**n) / (k2**m+((D0+D2)/2)**n) 

    # nonlinear cisinhibition 
    z += (1 - coupling_strength) * a2 * (k2**m / (k2**m + D**m))
    
    # sinusoidal driving term
    #z += a3 * np.sin(phi)
    
    # linear transactivation
    #z += coupling_strength * a2 * (D0+D2)/2
    
    # linear cisinhibition
    #z -= (1-coupling_strength) * a2 * D
    
    # degradation
    z -= g2 * H
    return z

def dphi_dt():
    return w

def Euler(x, dx_dt, dt):
    return x + dx_dt * dt

# simulate
def sim_lattice(d0, h0):# phi0):
    D = np.zeros((len(t), N_cells))
    H = np.zeros((len(t), N_cells))
    
    D[0] = d0
    H[0] = h0
    
    d0[0] = d0[1] = d0[-1] = d0[-2] = 0 
    h0[0] = h0[1] = h0[-1] = h0[-2] = 0
    
    for i in range(len(t) - 1):
        dd = dd_dt(H[i], D[i])
        dh = dh_dt(D[i], H[i])
        
        D[i+1] = Euler(D[i], dd, dt)
        H[i+1] = Euler(H[i], dh, dt)
    
    return D, H 

# initial conditions
d0 = np.random.rand(N_cells)
h0 = a3 + a3 * np.sin(k*x)

#d0 = np.array([i % 2 for i in range(44)])     checkerboard initialization

D, H = sim_lattice(d0, h0)

# plot
plt.figure(figsize=(10,6))
plt.plot(t, H, label="H", color='limegreen', alpha=0.4)
plt.plot(t, D, label="D", color='royalblue', alpha=0.4)
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title(f"Temporal behaviour Delta and Hes, coupling_strength={coupling_strength}")

plt.figure(figsize=(10,6))
plt.pcolor(D)
plt.title(f'Spatial behaviour Delta, coupling_strength={coupling_strength}')
plt.xlabel("Grid cells")
plt.ylabel("Time")

plt.figure(figsize=(10,6))
for i in range(2, N_cells-2):
    plt.scatter(D[:, i], H[:, i], c=t, s=2, cmap='plasma')
plt.xlabel("Delta (D)")
plt.ylabel("Hes (H)")
plt.title(f"Phase portrait, coupling strength={coupling_strength}")
plt.tight_layout()
plt.show()




