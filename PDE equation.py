# %%
###--- import packages ---###

import numpy as np
import matplotlib.pyplot as plt
from PDE_FIND import *
import pandas as pd
import matplotlib 
import os

from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape

import scipy.signal # for savitzky-golay filer


# %%
folder_path = r"C:\Users\Prince Savsaviya\Dropbox\My PC (LAPTOP-J9P1AHAD)\Downloads\Combined Dataset.csv"
discharge_capacity = {35:np.empty(0)}
df = pd.read_csv(folder_path)
x = np.array(df['Q discharge/mA.h']).reshape(1,-1)
y = np.array(df['cycle number']).reshape(-1,1)
discharge_capacity[35] = x
cycle = y

# %%
T = [35]
print(discharge_capacity[35].shape)
print(cycle.shape)

# %%
dcl = []
for i in range(len(discharge_capacity)):
    dcl_i = []
    for j in range(len(discharge_capacity[T[i]])):
        dcl_i.append(np.concatenate((discharge_capacity[T[i]][j], np.ones(len(cycle)-len(discharge_capacity[T[i]][j]))*discharge_capacity[T[i]][j][-1])))
    dcl.append(np.array(dcl_i))
dcl = np.array(dcl)
dcl.shape
## Convert the Redl data of each temperature to equal to time length (temperature, sample,time_value)

# %%
# averaging over all samples for each temperature condition
dcm = {}
for i in range(len(discharge_capacity)):
    dcm[T[i]] = np.mean(discharge_capacity[T[i]],axis=0)
# converting red colour vectors to same length - average
dcml = []
for i in range(len(dcm)):
    dcml.append(np.concatenate((dcm[T[i]], np.ones(len(cycle)-len(dcm[T[i]]))*dcm[T[i]][-1])))
dcml = np.array(dcml,dtype=object)
dcml.shape
## Convert the Redl data of each temperature to equal to time length (temperature,time_value)

"""
# averaging over all samples for each temperature condition
rm = {}
for i in range(len(DecData['red'])):
    rm[T[i]] = np.mean(DecData['red'][T[i]],axis=0)

# converting red colour vectors to same length - average
rml = []
for i in range(len(rm)):
    rml.append(np.concatenate((rm[T[i]], np.ones(len(DecData['time'])-len(rm[T[i]]))*rm[T[i]][-1])))
rml = np.array(rml)
rml.shape
## Convert the Redl data of each temperature to equal to time length (temperature,time_value)"""

# %%
###--- PDE-FIND parameters ---###
dt = cycle[1] - cycle[0] 
dx = 10 

P0=5 # max order of polynomial
D0=0 # only applicable for space derivative (not applicable in our case)
tol = 1 # threshold for STRidge
lam = 10**-5 # weight for regularization
method = 'FDConv' # method for numerical differentiation - Finite difference with smoothing

###--- odeint parameters ---###
atol0 = 10e-7
rtol0 = 10-10


# %%
# function to add Avrami terms of order n
def add_avrami(cycle, U, order, N):
    # initializations
    Rr = np.zeros((np.reshape(np.tile(np.power(cycle,order), (N, 2)),(len(cycle)*N,2), order = 'F')).shape)
    rhs_des = []
    
    # add t^n
    Rr[:,0] = np.reshape(np.tile(np.power(cycle,order), (N, 1)),(len(cycle)*N,), order = 'F')
    
    # add U*t^n
    Uu = np.reshape(np.tile(U, (N, 1)),(len(cycle)*N,), order = 'F')
    Rr[:,1] = np.reshape(np.multiply(Uu,Rr[:,0]),(-1,))
    
    # rhs descriptions
    rhs_des = ['t^'+str(order), 'U*t^'+str(order)]
    
    # 
    return Rr, rhs_des


# %%
pde_data = np.vstack(dcl[:]) # create a large matrix of all samples and all temperatures
pde_data.shape

# %%
Ut,R,rhs_des = build_linear_system(pde_data, dt, dx, D=D0, P=P0, time_diff = method, space_diff = method) #build library of polynomials -built-in in PDE-FIND
print(['1'] + rhs_des[1:])

# %%
###--- Add sine and cosine ---###
# add sin
R = np.append(R,np.reshape(np.sin(R[:,1]),(-1,1)),1)
rhs_des = rhs_des + ['sin(u)']

# add cos
R = np.append(R,np.reshape(np.cos(R[:,1]),(-1,1)),1)
rhs_des = rhs_des + ['cos(u)']
print(['1'] + rhs_des[1:])


# %%
###--- Add Avrami terms ---###

n_avrami = [0.5,1, 2, 3]

for order in n_avrami:
    # add avami terms
    Rr, rhs_add = add_avrami(np.tile(cycle, (len(dcl)*len(dcl[0]), 1)), np.real(R[:,1]), order, 1) #compute terms
    R = np.append(R,Rr,1)
    rhs_des = rhs_des + rhs_add
print(['1'] + rhs_des[1:])

# %%
###--- Add exp(-n/T) and T ---###

# Temperature Terms
Tvec = []
for i in range(len(T)):
    Tvec.append(np.multiply(T[i]+273,np.ones((len(dcl[i])*len(cycle),))))
Tvec1 = np.reshape(Tvec, (len(dcl)*len(dcl[0])*len(cycle),), order = 'F')

expTvec = np.exp(np.divide(-100,Tvec))

# add T
R = np.append(R,np.reshape(Tvec,(-1,1)),1)
rhs_des = rhs_des + ['T']

# add exp(-100/T)
R = np.append(R,np.reshape(expTvec,(-1,1)),1)
rhs_des = rhs_des + ['exp(-100/T)']

print(['1'] + rhs_des[1:])

# %%
w = np.real(TrainSTRidge(R,Ut,lam,tol,normalize = 2))
print_pde(w, rhs_des)

# %%
###--- Function to estimate derivative from the equation identified by PDE-FIND ---###
def pdediffeq(u,t,T,coefs):
    
    if type(t) == float:
        n = 1
    else:
        n = len(t)
    
    dudt =  coefs[0] + np.multiply(coefs[1],u) \
    +np.multiply(coefs[2],np.power(u,2)) + np.multiply(coefs[3],np.power(u,3))+\
    np.multiply(coefs[4],np.power(u,4))+ np.multiply(coefs[5],np.power(u,5)) +\
    np.multiply(coefs[6],np.sin(u)) + np.multiply(coefs[7],np.cos(u)) +\
    np.multiply(coefs[8],np.power(t,n_avrami[0]).reshape(-1))+ np.multiply(coefs[9],np.multiply(u,np.power(t,n_avrami[0]).reshape(-1))) + \
    np.multiply(coefs[10],np.power(t,n_avrami[1]).reshape(-1))+ np.multiply(coefs[11],np.multiply(u,np.power(t,n_avrami[1]).reshape(-1))) + \
    np.multiply(coefs[12],np.power(t,n_avrami[2]).reshape(-1))+ np.multiply(coefs[13],np.multiply(u,np.power(t,n_avrami[2]).reshape(-1))) + \
    np.multiply(coefs[14],np.power(t,n_avrami[3]).reshape(-1))+ np.multiply(coefs[15],np.multiply(u,np.power(t,n_avrami[3]).reshape(-1))) + \
    np.multiply(coefs[16]*(T+273),np.ones((n,))) + np.multiply(coefs[17]*np.exp(-100/(T+273)),np.ones((n,)))

    dudt = np.reshape(dudt,(-1,))
    dudt = dudt - dudt[-1]
    
    return dudt

# %%
###--- function to choose best initial condition for integration ---###
from scipy.optimize import fmin
def bestUo(func, solavg, cycle, T, i):
    def fit_mse(U0):
        curve = np.reshape(odeint(func, U0, np.reshape(cycle,(-1)), args=(T,solavg,), atol = atol0, rtol = rtol0), (len(cycle)))
        return mse(dcml[i],curve)
    
    U0 = dcml[i][0]
    Uopt = fmin(fit_mse, U0, xtol=1e-3, disp=False)
    return Uopt

# %%
mae_alld = [] # vector to store the MAE between actual derivative and PDE-FIND estimate

for i in range(len(T)):
    fig, ax = plt.subplots(1, 1, figsize = (3,2), dpi=150)
    ax.tick_params(axis='both', which='major', pad=1)
    #plt.title("Temperature="+str(T[i]) + "$^\circ$C", fontsize = 10)
    
    # estimate actual derivative through PDE-FIND
    pde_data = np.tile(dcml[i][:],(1,1))
    Ut_actual,_,_ = build_linear_system(pde_data, dt, dx, D=D0, P=P0, time_diff = method, space_diff = method)
    
    # First, the actual derivative needs to be smoothened. we use the savitzky-golay filter
    Ut_smooth = scipy.signal.savgol_filter(Ut_actual.reshape((-1)), 100, 5) 
    
    # PDE-FIND estimate
    Ut_pdefind = pdediffeq(dcl[i,-1][:], cycle, T[i], w) 
    
    
    plt.plot(cycle[:],Ut_actual[:], label = 'Actual -Noisy')
    plt.plot(cycle[:], Ut_smooth[:], label = 'Actual -Smooth')
    plt.plot(cycle[:], Ut_pdefind[:], label = 'PDE-FIND')
    
    plt.xlabel("Cycle", fontsize = 10, labelpad = 0)
    plt.ylabel("$dU/dt$", fontsize = 10, labelpad = 0)
    plt.legend(fontsize = 8)
    # calculate MAE error
    mae_alld.append(mape(Ut_smooth,Ut_pdefind))
    plt.legend(fontsize = 8,bbox_to_anchor=(1.0, 1.0), loc='upper left')


# %%
fig, ax = plt.subplots(1, 1, figsize = (3,2))
ax.tick_params(axis='both', which='major', pad=1)
plt.bar(range(len(T)),np.divide(mae_alld,1),
        width = 0.8,
        color = 'b',
        align ='center',
        alpha=0.5,
        ecolor='black',
        capsize=14)

ax.set_ylabel('MAPE-Derivative', fontsize = 10, labelpad = 0)
ax.set_xlabel('Temperature', fontsize = 10, labelpad = 0)

ax.set_xticks(range(len(T)))
ax.set_xticklabels([str(T[i])+'$^\circ$C' for i in range(len(T))], fontsize = 9)
ax.set_title('')
ax.yaxis.grid(True)
plt.tight_layout()

# %%

mae_alli = []

for i in range(len(T)):
    fig, ax = plt.subplots(1, 1, figsize = (3,2), dpi=150)
    ax.tick_params(axis='both', which='major', pad=1)
    plt.title("Temperature="+str(T[i]) + "$^\circ$C", fontsize = 10)
    
    # estimate actual derivative through PDE-FIND
    pde_data = np.tile(dcml[i][:],(1,1))
    Ut_actual,_,_ = build_linear_system(pde_data, dt, dx, D=D0, P=P0, time_diff = method, space_diff = method)
    
    # First, the actual derivative needs to be smoothened. we use the savitzky-golay filter
    Ut_smooth = scipy.signal.savgol_filter(Ut_actual.reshape((-1)), 100, 5) 
    
    # PDE-FIND estimate
    Ut_pdefind = pdediffeq(dcl[i,-1][:], cycle, T[i], w)
    
    # integrate PDE-FINd equation with odeint
    U0 = bestUo(pdediffeq, w, cycle, T[i], i ) # inital conditions at t=0
    print(U0)
    curve_pdefind = np.reshape(odeint(pdediffeq, U0, np.reshape(cycle[:],(-1)), args=(T[i],w,)), (len(cycle[:])))
    plt.title("Temperature="+str(T[i])+ "$^\circ$C", fontsize = 10)
    plt.plot(cycle[:],dcml[i][:], label = "Experimental data")
    plt.plot(cycle[:],curve_pdefind[:], label="PDE-FIND")
    plt.ylabel("Discharge Capacity", fontsize = 10, labelpad = 0)
    plt.xlabel("cycle", fontsize = 10, labelpad = 0)
    plt.legend(fontsize = 8)
    
    plt.tight_layout()
    
    #calculate MAE error
    mae_alli.append(mape(dcml[i],curve_pdefind))

# %%
###--- bar-plot: comparing the actual data with PDE-FIND's estimate of integrated curve ---###

fig, ax = plt.subplots(1, 1, figsize = (3,2))
ax.tick_params(axis='both', which='major', pad=1)
plt.bar(range(len(T)),np.divide(mae_alli,1),
        width = 0.8,
        color = 'b',
        align ='center',
        alpha=0.5,
        ecolor='black',
        capsize=14)

ax.set_ylabel('MAPE-Integrated', fontsize = 10, labelpad = 0)
ax.set_xlabel('Temperature', fontsize = 10, labelpad = 0)

ax.set_xticks(range(len(T)))
ax.set_xticklabels([str(T[i])+'$^\circ$C' for i in range(len(T))], fontsize = 9)
ax.set_title('')
ax.yaxis.grid(True)
plt.tight_layout()


# %%


# %%



