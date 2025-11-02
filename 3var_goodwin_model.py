# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 13:21:09 2025

@author: Lotte
"""

import matplotlib.pyplot as plt
import numpy as np

# define constants
a1, a21, a22, a3, a4 = 5,5,5,5,5
g1, g2, g3 = 0.5,0.5,0.5
k1, k2, k3 = 1,1,1
n, m = 10,10
coupling_strength = 1

period = 0.1 
w =  2*np.pi/period
t_final = 100
dt = 0.1
t = np.arange(0,t_final,dt)
N_cells = 44

# model
def dd_dt(Hp,D):
    x = (a1 * k1**n)/(k1**n + Hp**n)    #Hes protein inhibition
    x -= g1*D                           #degredation
    return x

def dhm_dt(D,Hm,Hp, t, theta): 
    D2 = np.roll(D, 1)
    D0 = np.roll(D,-1)
    
    #y = (a4 * k1**n)/(k1**n + Hp**n)                           #autoinhibition
    y = a21*coupling_strength*(D2+D0)/2                         #linear transactivation
    #y = coupling_strength*a21*(k3**n/((D0 + D2)**n+k3**n))     #nonlinear transactivation
    y += a22*(1-coupling_strength)*D                           #linear cisactivation
    #y += (1-coupling_strength)*((a22 * k2**n)/(k2**n + D**n))   #cisinhibition
    #y += a3*np.sin(w*t + theta)                                #sinusoidal driving force
    y -= g2*Hm                                                  #degradation
    return y

def dhp_dt(Hm,Hp,t,theta): 
    z = a3*Hm                       #linear Hes mRNA activation
    #z = a3*(k3**n/(Hm**n+k3**n))   #nonlinear Hes mRNA activation
    #z *= np.sin(w*t + theta)        #forcing Hes stimulation into sine
    z -= g3*Hp                      #degredation
    return z

# simulate
def Euler(x, dx_dt, dt):
    x_new = x + dx_dt*dt    
    return x_new

def sim_lattice(d0, hm0, hp0):
    D = np.zeros((len(t),N_cells))
    Hm = np.zeros((len(t),N_cells))
    Hp = np.zeros((len(t),N_cells))
    theta = np.zeros(N_cells)
        
    D[0,:] = d0
    Hm[0,:] = hm0
    Hp[0,:] = hp0
    
    for i, ti in enumerate(t[:-1]):
        dd = dd_dt(hp0,d0)
        dhm = dhm_dt(d0,hm0,hp0, ti*dt, period/2) 
        dhp = dhp_dt(hm0,hp0, ti*dt, theta)
        
        d0 = Euler(d0,dd,dt)
        hm0 = Euler(hm0,dhm,dt)
        hp0 = Euler(hp0,dhp,dt)   
       
        d0[0] = d0[1] = d0[-1] = d0[-2] = 0 
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
plt.plot(t, Hp, label="Hes protein", color='limegreen', alpha = 0.5)
plt.plot(t, D, label="Delta", color='royalblue', alpha = 0.5)
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title(f"Temporal oscillations, coupling_strength={coupling_strength}")

plt.figure(figsize=(10,6))
plt.pcolor(D)
plt.title(f'Spatial oscillation Delta, Hes period={period}, coupling_strength={coupling_strength}')
plt.xlabel("Grid cells")
plt.ylabel("Time")



