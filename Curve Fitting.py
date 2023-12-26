# %%
###--- import packages ---###
import numpy as np
import matplotlib.pyplot as plt
#from PDE_FIND import *
import pandas as pd
import matplotlib 
import os

from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
import scipy.signal

# %% [markdown]
# # 1st degree Polynomial

# %%
folder_path = r"C:\Users\Prince Savsaviya\Dropbox\My PC (LAPTOP-J9P1AHAD)\Downloads\Dataset_1_NCA_battery\Dataset_1_NCA_battery\update_data\selected data test"
files = sorted(os.listdir(folder_path))
# Define the polynomial function
def polynomial(x, a, b, c, d, e, f):
    return d*x+ c
for file in files:
    df = pd.read_csv(os.path.join(folder_path,file))
    y = np.array(df['Q discharge/mA.h'])
    x = np.array(df['cycle number'],dtype='int32')

    # Perform the curve fitting
    params, _ = curve_fit(polynomial, x, y)

    # Generate the fitted curve
    x_fit = np.linspace(0, len(y), 500)
    y_fit = polynomial(x_fit, *params)
    print(params)
    # Plot the data and the fitted curve
    fig,ax = plt.subplots(figsize=(8, 6),edgecolor='black')
    #fig = plt.figure(figsize=(8, 6),edgecolor='black')
    plt.scatter(x, y, label='Actual')
    plt.plot(x_fit, y_fit, 'r-', label='Fitted')
    plt.xlabel('Cycle Number',fontsize=15,fontweight='bold')
    plt.ylabel('Q discharge/mA.h',fontsize=15,fontweight='bold')
    plt.legend(fontsize=15)
    plt.legend().set_frame_on(False)
    plt.grid(False)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    ax.spines[:].set_linewidth(3)
    #plt.savefig(f"C:\\Users\\Prince Savsaviya\\Desktop\\Graphs\\{file}.jpeg",dpi=3000,format="jpeg", bbox_inches='tight')
    plt.show()

# %% [markdown]
# # 2nd degree Polynomial

# %%
folder_path = r"C:\Users\Prince Savsaviya\Dropbox\My PC (LAPTOP-J9P1AHAD)\Downloads\Dataset_1_NCA_battery\Dataset_1_NCA_battery\update_data\selected data test"
files = sorted(os.listdir(folder_path))
# Define the polynomial function
def polynomial(x, a, b, c, d, e, f):
    return b*x**2+d*x+ c
for file in files:
    df = pd.read_csv(os.path.join(folder_path,file))
    y = np.array(df['Q discharge/mA.h'])
    x = np.array(df['cycle number'],dtype='int32')

    # Perform the curve fitting
    params, _ = curve_fit(polynomial, x, y)

    # Generate the fitted curve
    x_fit = np.linspace(0, len(y), 500)
    y_fit = polynomial(x_fit, *params)
    print(params)
    # Plot the data and the fitted curve
    fig,ax = plt.subplots(figsize=(8, 6),edgecolor='black')
    #fig = plt.figure(figsize=(8, 6),edgecolor='black')
    plt.scatter(x, y, label='Actual')
    plt.plot(x_fit, y_fit, 'r-', label='Fitted')
    plt.xlabel('Cycle Number',fontsize=15,fontweight='bold')
    plt.ylabel('Q discharge/mA.h',fontsize=15,fontweight='bold')
    plt.legend(fontsize=15)
    plt.legend().set_frame_on(False)    
    plt.grid(False)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    ax.spines[:].set_linewidth(3)
    #plt.savefig(f"C:\\Users\\Prince Savsaviya\\Desktop\\Graphs\\{file}.jpeg",dpi=3000,format="jpeg", bbox_inches='tight')
    plt.show()

# %% [markdown]
# # 3rd degree Polynomail

# %%
folder_path = r"C:\Users\Prince Savsaviya\Dropbox\My PC (LAPTOP-J9P1AHAD)\Downloads\Dataset_1_NCA_battery\Dataset_1_NCA_battery\update_data\selected data test"
files = sorted(os.listdir(folder_path))
i = 0
l=[]
# Define the polynomial function
def polynomial(x, a, b, c, d, e, f):
    return c*x**3 + d*x**2 + e*x + f
for file in files:
    df = pd.read_csv(os.path.join(folder_path,file))
    y = np.array(df['Q discharge/mA.h'])
    x = np.array(df['cycle number'],dtype='int32')

    # Perform the curve fitting
    params, _ = curve_fit(polynomial, x, y)

    # Generate the fitted curve
    x_fit = np.linspace(0, len(y), 500)
    y_fit = polynomial(x_fit, *params)
    print(params)
    # Plot the data and the fitted curve
    fig,ax = plt.subplots(figsize=(8, 6),edgecolor='black')
    #fig = plt.figure(figsize=(8, 6),edgecolor='black')
    plt.scatter(x, y, label='Actual')
    plt.plot(x_fit, y_fit, 'r-', label='Fitted')
    plt.xlabel('Cycle Number',fontsize=15,fontweight='bold')
    plt.ylabel('Q discharge/mA.h',fontsize=15,fontweight='bold')
    plt.legend(fontsize=15)
    plt.legend().set_frame_on(False)
    plt.grid(False)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    ax.spines[:].set_linewidth(3)
    #plt.savefig(f"C:\\Users\\Prince Savsaviya\\Desktop\\Graphs\\{file}.jpeg",dpi=3000,format="jpeg", bbox_inches='tight')
    plt.show()
    

# %% [markdown]
# # 4th degree Polynomial

# %%
folder_path = r"C:\Users\Prince Savsaviya\Dropbox\My PC (LAPTOP-J9P1AHAD)\Downloads\Dataset_1_NCA_battery\Dataset_1_NCA_battery\update_data\selected data test"
files = sorted(os.listdir(folder_path))
i = 0
l=[]
# Define the polynomial function
def polynomial(x, a, b, c, d, e, f):
    return b*x**4 + c*x**3 + d*x**2 + e*x + f
for file in files:
    df = pd.read_csv(os.path.join(folder_path,file))
    y = np.array(df['Q discharge/mA.h'])
    x = np.array(df['cycle number'],dtype='int32')

    # Perform the curve fitting
    params, _ = curve_fit(polynomial, x, y)

    # Generate the fitted curve
    x_fit = np.linspace(0, len(y), 500)
    y_fit = polynomial(x_fit, *params)
    print(params)
    # Plot the data and the fitted curve
    fig,ax = plt.subplots(figsize=(8, 6),edgecolor='black')
    #fig = plt.figure(figsize=(8, 6),edgecolor='black')
    plt.scatter(x, y, label='Actual')
    plt.plot(x_fit, y_fit, 'r-', label='Fitted')
    plt.xlabel('Cycle Number',fontsize=15,fontweight='bold')
    plt.ylabel('Q discharge/mA.h',fontsize=15,fontweight='bold')
    plt.legend(fontsize=15)
    plt.legend().set_frame_on(False)
    plt.grid(False)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    ax.spines[:].set_linewidth(3)
    #plt.savefig(f"C:\\Users\\Prince Savsaviya\\Desktop\\Graphs\\{file}.jpeg",dpi=3000,format="jpeg", bbox_inches='tight')
    plt.show()
    

# %% [markdown]
# # 5th degree Polynomial

# %%
folder_path = r"C:\Users\Prince Savsaviya\Dropbox\My PC (LAPTOP-J9P1AHAD)\Downloads\Dataset_1_NCA_battery\Dataset_1_NCA_battery\update_data\selected data test"
files = sorted(os.listdir(folder_path))
i = 0
l=[]
# Define the polynomial function
def polynomial(x, a, b, c, d, e, f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
for file in files:
    df = pd.read_csv(os.path.join(folder_path,file))
    y = np.array(df['Q discharge/mA.h'])
    x = np.array(df['cycle number'],dtype='int32')

    # Perform the curve fitting
    params, _ = curve_fit(polynomial, x, y)

    # Generate the fitted curve
    x_fit = np.linspace(0, len(y), 500)
    y_fit = polynomial(x_fit, *params)
    print(params)
    # Plot the data and the fitted curve
    fig,ax = plt.subplots(figsize=(8, 6),edgecolor='black')
    #fig = plt.figure(figsize=(8, 6),edgecolor='black')
    plt.scatter(x, y, label='Actual')
    plt.plot(x_fit, y_fit, 'r-', label='Fitted')
    plt.xlabel('Cycle Number',fontsize=15,fontweight='bold')
    plt.ylabel('Q discharge/mA.h',fontsize=15,fontweight='bold')
    plt.legend(fontsize=15)
    plt.legend().set_frame_on(False)
    plt.grid(False)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    ax.spines[:].set_linewidth(3)
    #plt.savefig(f"C:\\Users\\Prince Savsaviya\\Desktop\\Graphs\\{file}.jpeg",dpi=3000,format="jpeg", bbox_inches='tight')
    plt.show()
    

# %%



