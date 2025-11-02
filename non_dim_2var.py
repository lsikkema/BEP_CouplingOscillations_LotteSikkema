# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 16:17:55 2025

@author: Lotte
"""

import matplotlib.pyplot as plt
import numpy as np

# define constants
ag = 2
k= 1
n = 10

t_final = 800
dt = 0.1
t = np.arange(0,t_final,dt)
N_cells = 44
period = 1000 #120*60 #sec
w =  2*np.pi/period

a3g =1
coupling_strength = 1

#model
def dd_dt(H,D):
    y = ag*(k**n/(H**n+k**n)) #nonlinear inhibition
    y -= D
    return y

def dh_dt(D,H,t, theta): #add theta
    D2 = np.roll(D, 1)
    D0 = np.roll(D,-1)
    
    z = a3g*np.sin(w*t+theta) #sinusoidal driving
    z += coupling_strength*ag*(D0+D2)/2 #linear transactivation
    z += (1-coupling_strength)*ag*(k**n/(k**n + D**n)) #cisinhibition
    z -= H
    return z

#simulate
def Euler(x, dx_dt, dt):
    x_new = x + dx_dt*dt    
    return x_new

def sim_lattice(d0,h0):
    D = np.zeros((len(t),N_cells))
    H = np.zeros((len(t),N_cells))
    #theta = np.linspace(0,0.01,N_cells)
    theta = np.zeros(N_cells)
    
     
    D[0,:] = d0
    H[0,:] = h0
    
    
    for i, ti in enumerate(t[:-1]):
        dd = dd_dt(h0,d0)
        dh = dh_dt(d0,h0, ti*dt, theta) #add theta
        
        d0 = Euler(d0,dd,dt)
        h0 = Euler(h0,dh,dt)   
       
        d0[0] = d0[1] = d0[-1] = d0[-2] = 0 #onthou dat er dus 4 extra cellen zijn
        h0[0] = h0[1] = h0[-1] = h0[-2] = 0
        
        D[i,:] = d0
        H[i,:] = h0

    return D,H

# initial values
#d0 = np.linspace(0,2,N_cells) 
d0=np.random.rand(N_cells)*2
h0= np.random.rand(N_cells)
#h0=np.linspace(0,2,N_cells) 
#d0=h0=np.zeros(N_cells)

D,H = sim_lattice(d0,h0)

# plot
plt.figure(figsize=(10,6))
plt.plot(t, H, label="h", color='crimson', alpha = 0.3)
plt.plot(t, D, label="d", color='royalblue', alpha = 0.5)
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Temporal oscillations of Delta and Hes")

plt.figure(figsize=(10,6))
plt.plot(t, D[:,2], label="Delta, cell 1", color='royalblue', alpha = 0.5)
plt.plot(t, H[:,2], label="Hes, cell 1", color='crimson', alpha = 0.5)
plt.plot(t, D[:,3], label="Delta, cell 2", color='cornflowerblue', alpha = 0.5)
plt.plot(t, H[:,3], label="Hes, cell 2", color='indianred', alpha = 0.5)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Temporal oscillations")


plt.figure(figsize=(10,6))
plt.pcolor(D)
plt.title('Spatial oscillations of Delta')
plt.xlabel("Grid cells")
plt.ylabel("Time")

'''
plt.figure(figsize=(10,6))
plt.plot(D[:,10],H[:,10], color='royalblue', alpha = 0.5)
plt.title('Limit cycle Hes/Delta')
plt.xlabel("D")
plt.ylabel("H")
plt.xlim(0,2)
plt.ylim(0,50)
'''




