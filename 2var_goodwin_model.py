# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:19:01 2025

@author: Lotte
"""
import matplotlib.pyplot as plt
import numpy as np

# define constants
a1, a21, a22, a3 = 1,1,1,0.1
g1, g2 = 0.5,0.5
k1, k2, k3, k4 = 1,1,1,1

t_final = 100
dt = 0.1
t = np.arange(0,t_final,dt)
N_cells = 104

period = 1
w =  2*np.pi/period

coupling_strength = 0.5
n = 10
m = 10

# model
def dd_dt(H,D):
    y = a1*(k1**n/(H**n+k1**n))     #nonlinear inhibition
    y -= g1*D                       #degradation
    return y

def dh_dt(D,H,t, theta): 
    D2 = np.roll(D, 1)
    D0 = np.roll(D,-1)

    z = a3*np.sin(w*t+theta)                                            #sinusoidal driving
    z += coupling_strength*a21*(D0+D2)/2                                #linear transactivation
    #z += coupling_strength*a21*(k3**n/((D0 + D2)/2**(-1*n)+k3**n))     #nonlinear transactivation
    #z += (1-coupling_strength)*a22*D                                   #linear cisactivation
    z += (1-coupling_strength)*a22*(k4**m/(k4**m + D**m))               #cisinhibition
    z -= g2*H                                                           #degradation
    return z

# simulate
def Euler(x, dx_dt, dt):
    x_new = x + dx_dt*dt    
    return x_new

def sim_lattice(d0,h0):
    D = np.zeros((len(t),N_cells))
    H = np.zeros((len(t),N_cells))
    theta = np.zeros(N_cells)
        
    D[0,:] = d0
    H[0,:] = h0
    
    for i, ti in enumerate(t[:-1]):
        dd = dd_dt(h0,d0)
        dh = dh_dt(d0,h0, ti*dt, theta) 
        
        d0 = Euler(d0,dd,dt)
        h0 = Euler(h0,dh,dt)   
       
        d0[0] = d0[1] = d0[-1] = d0[-2] = 0 
        h0[0] = h0[1] = h0[-1] = h0[-2] = 0
        
        D[i,:] = d0
        H[i,:] = h0

    return D,H

# initial values
d0 = np.random.rand(N_cells)
h0 = np.random.rand(N_cells)


D,H = sim_lattice(d0,h0)

# plot
plt.figure(figsize=(10,6))
plt.plot(t, H, label="h", color='limegreen', alpha = 0.3)
plt.plot(t, D, label="d", color='royalblue', alpha = 0.5)
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title(f"Temporal oscillations of Delta and Hes, coupling strength = {coupling_strength}")

plt.figure(figsize=(10,6))
plt.pcolor(D)
plt.title(f'Spatial oscillations of Delta, coupling strength = {coupling_strength}')
plt.xlabel("Grid cells")
plt.ylabel("Time")

cells_to_plot = [3, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]

plt.figure(figsize=(10,6)) 
for cell in cells_to_plot:
    plt.plot(D[:,cell], H[:,cell], alpha=0.5)
plt.xlabel("Delta (D)")
plt.ylabel("Hes (H)")
plt.title(f"Limit cycles in t–H phase space, coupling strength = {coupling_strength}")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
plt.plot(t, D[:,2], label="Delta, cell 1", color='royalblue', alpha = 0.5)
plt.plot(t, H[:,2], label="Hes, cell 1", color='crimson', alpha = 0.5)
plt.plot(t, D[:,3], label="Delta, cell 2", color='cornflowerblue', alpha = 0.5)
plt.plot(t, H[:,3], label="Hes, cell 2", color='indianred', alpha = 0.5)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Temporal oscillations")




