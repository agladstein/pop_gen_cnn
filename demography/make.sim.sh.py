from random import *
import json
import numpy as np
import subprocess
from sys import argv

def scale_N(N0, N):
    N_scaled = N / N0
    return N_scaled

def scale_T(N0, T):
    T_scaled = T/(4*N0)
    return T_scaled

def run_sim(ms_command):
    subprocess.call(ms_command.split())

sim_id = argv[1]
    
length = 1500000
mu = 1.2*10**(-8)/10
r = 1*10**(-8)
N0 = np.random.uniform(100, 40000)
T1 = np.random.uniform(1, 3500)
N1 = np.random.uniform(100, 5000)
T2 = np.random.uniform(1, 3500) + T1
N2 = np.random.uniform(100, 20000)

params_file = open("params/true.params.{}.json".format(sim_id),"w+")
params_file.write(str(N0)+"\n")
params_file.write(str(T1)+"\n")
params_file.write(str(N1)+"\n")
params_file.write(str(T2)+"\n")
params_file.write(str(N2)+"\n")
params_file.close()

theta = 4*N0*mu*length
rho = 4*N0*r*length
N0_scaled = scale_N(N0, N0)
T1_scaled = scale_T(N0, T1)
N1_scaled = scale_N(N0, N1)
T2_scaled = scale_T(N0, T2)
N2_scaled = scale_N(N0, N2)

ms_command = 'ms 100 1 -t {} -r {} {} -eN {} {} -eN {} {}'.format(theta, rho, length, T1_scaled, N1_scaled, T2_scaled, N2_scaled)
print(ms_command)
#TODO: print ms_command to file
command_file = open("demography.sims.txt","a+")
command_file.write(str(ms_command))
command_file.close()

run_sim(ms_command)
