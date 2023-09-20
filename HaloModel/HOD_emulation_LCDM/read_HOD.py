import numpy as np 
import h5py 
import time 
from pathlib import Path

import pandas as pd

"""
Not important, just for testing.
Prints various properties of the HOD catalogues.
E.g. test that the number of galaxies is correctly constrained,
print the HOD parameters, etc.
"""



HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
HALO_ARRAYS_PATH = f"{HOD_DATA_PATH}/version0"

OUTFILEPATH = f"{HOD_DATA_PATH}/HOD_data"
dataset_names = ['train', 'test', 'val']

def read_hdf5_files(ng_fixed=False):
    ng_list = []
    nc_list = []
    ns_list = []

    for flag in dataset_names:
        # print(f"flag={flag}") 
        if ng_fixed:
            halocat_fname = f"halocat_{flag}_ng_fixed"
            node_params_fname = f"HOD_parameters_ng_fixed_{flag}"
        else:
            halocat_fname = f"halocat_{flag}"
            node_params_fname = f"HOD_parameters_{flag}"

        fff = h5py.File(f"{OUTFILEPATH}/{halocat_fname}.hdf5", "r")
        node_params_df = pd.read_csv(f"{HOD_DATA_PATH}/HOD_data/{node_params_fname}.csv")

        N_nodes = len(fff.keys())
        ng_flag = []
        nc_flag = []
        ns_flag = []


        for node_idx in range(N_nodes):

            HOD_group = fff[f"node{node_idx}"] 

            ng_flag.append(HOD_group.attrs['ng'])
            nc_flag.append(HOD_group.attrs['nc'])
            ns_flag.append(HOD_group.attrs['ns'])
            
        ng_list.append(ng_flag)
        nc_list.append(nc_flag)
        ns_list.append(ns_flag)
        # print(ng_flag)
        fff.close()

        # print("         ng       | nc | ns")
        # for i in range(len(ng_flag)):
            # print(f"node{i}: ng={ng_flag[i]:.4e} | nc={nc_flag[i]:.4e} | ns={ns_flag[i]:.4e}")

    # print("ng_list:", ng_list)
    # print("nc_list:", nc_list)
    # print("ns_list:", ns_list)
    print(f"ng mean +/- std: ")
    print(f" train: {np.mean(ng_list[0]):.4e} +/- {np.std(ng_list[0]):.4e}")
    print(f" test:  {np.mean(ng_list[1]):.4e} +/- {np.std(ng_list[1]):.4e}")
    print(f" val:   {np.mean(ng_list[2]):.4e} +/- {np.std(ng_list[2]):.4e}")

def print_HOD_parameters(flag, ng_fixed=True, print_every_n=5):
    HOD_params_path = OUTFILEPATH
    if ng_fixed:
        outfile = f"{HOD_params_path}/HOD_parameters_ng_fixed_{flag}.csv"
    else:
        outfile = f"{HOD_params_path}/HOD_parameters_{flag}.csv"
    df = pd.read_csv(outfile)
    print("     sigma_logM | alpha  | kappa  | log10M1 ")
    for i in range(len(df)):
        sigma_logM = df["sigma_logM"][i]
        alpha = df["alpha"][i]
        kappa = df["kappa"][i]
        log10M1 = df["log10M1"][i]
        print(f"node{i}: {sigma_logM:8.4f} | {alpha:.4f} | {kappa:.4f} | {log10M1:.4f}")
        if (i+1) % print_every_n == 0:
            input()
            print("     sigma_logM | alpha  | kappa  | log10M1 ")



def read_csv_original():
    HOD = pd.read_csv(f"../example/v130_HOD.csv")
    N_central = len(HOD[HOD["galaxy_type"] == "central"])
    N_satellite = len(HOD[HOD["galaxy_type"] == "satellite"])
    N_total = len(HOD)
    # print(HOD.value_counts("galaxy_type"))
    print(f"Number of centrals: {N_central}")
    print(f"Number of satellites: {N_satellite}")
    print(f"Total number of galaxies: {N_total}")
    
    ng = N_total / 2000.0**3
    print(f"galaxy number density: {ng:.4e}")


def read_HOD_fiducial_hdf5():
    fff = h5py.File(f"{OUTFILEPATH}/halocat_fiducial.hdf5", "r")

    ng = fff.attrs['ng']
    nc = fff.attrs['nc']
    ns = fff.attrs['ns']
    print(f"ng: {ng:.4e}")
    print(f"nc: {nc:.4e}")
    print(f"ns: {ns:.4e}")


print("No constraints on ng:")
read_hdf5_files()
print()
print("Constraints on ng:")
read_hdf5_files(ng_fixed=True)
# read_csv_original()
# read_HOD_fiducial_hdf5()

# print_HOD_parameters("train", ng_fixed=True)