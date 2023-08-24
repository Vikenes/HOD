import numpy as np 
import h5py 
import time 
# from dq import Cosmology
# import hmd # No GPU/TPU found 
# from hmd.catalogue import ParticleCatalogue, HaloCatalogue, GalaxyCatalogue
# from hmd.occupation import Zheng07Centrals, Zheng07Sats
# from hmd.galaxy import Galaxy
# from hmd.profiles import FixedCosmologyNFW # Initialize SigmaM emulator 
# from hmd.populate import HODMaker
from pathlib import Path

import pandas as pd

# import warnings
# warnings.filterwarnings("ignore")

HOD_DATA_PATH = "/mn/stornext/d5/data/vetleav/HOD_AbacusData/c000_LCDM_simulation"
HALO_ARRAYS_PATH = f"{HOD_DATA_PATH}/version0"

OUTFILEPATH = f"{HOD_DATA_PATH}/HOD_data"
dataset_names = ['train', 'test', 'val']

def read_hdf5_files():
    ng_list = []
    nc_list = []
    ns_list = []

    for flag in dataset_names:
        print(f"flag={flag}") 
        fff = h5py.File(f"{OUTFILEPATH}/halocat_{flag}.hdf5", "r")
        node_params_df = pd.read_csv(f"{HOD_DATA_PATH}/HOD_data/HOD_parameters_{flag}.csv")

        N_nodes = len(fff.keys())
        ng_flag = []
        nc_flag = []
        ns_flag = []

        n1 = fff['node0']
      

        for node_idx in range(N_nodes):
            # print(f"Running node{node_idx}...", end=" ")

            HOD_group = fff[f"node{node_idx}"] 

            ng_flag.append(HOD_group.attrs['ng'])
            nc_flag.append(HOD_group.attrs['nc'])
            ns_flag.append(HOD_group.attrs['ns'])
            
        ng_list.append(ng_flag)
        nc_list.append(nc_flag)
        ns_list.append(ns_flag)
        # print(ng_flag)
        fff.close()

        print("         ng       |log10Mmin|sigma_logM|log10M1|kappa|alpha")
        for i in range(len(ng_flag)):
            logM_min    = node_params_df['log10Mmin'].iloc[i]
            sigma_logM  = node_params_df['sigma_logM'].iloc[i]
            logM1       = node_params_df['log10M1'].iloc[i]
            kappa       = node_params_df['kappa'].iloc[i]
            alpha       = node_params_df['alpha'].iloc[i]
            print(f"node{i}: {ng_flag[i]:.4e} | {logM_min:.4f} | {sigma_logM:.4f} | {logM1:.4f} | {kappa:.4f} | {alpha:.4f}")
        exit()

    # print("ng_list:", ng_list)
    # print("nc_list:", nc_list)
    # print("ns_list:", ns_list)
    print(f"ng mean +/- std: ")
    print(f" train: {np.mean(ng_list[0]):.4e} +/- {np.std(ng_list[0]):.4e}")
    print(f" test:  {np.mean(ng_list[1]):.4e} +/- {np.std(ng_list[1]):.4e}")
    print(f" val:   {np.mean(ng_list[2]):.4e} +/- {np.std(ng_list[2]):.4e}")



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



def make_csv_files():
    for flag in dataset_names:
        ### Create csv file from hdf5 file 
        
        # file_pk_h5py = h5py.File(DATAPATH + f'Pk_{flag}.hdf5', 'r')
        fff = h5py.File(OUTFILEPATH + f"/HOD_{flag}.hdf5", "r")


        _lst = []
        for key in fff.keys():

            _lst.append(pd.DataFrame({
                'log10Mmin' : fff[key].attrs["log10Mmin"], 
                'sigma_logM': fff[key].attrs["sigma_logM"],    
                'log10M1'   : fff[key].attrs["log10M1"],   
                'kappa'     : fff[key].attrs["kappa"], 
                'alpha'     : fff[key].attrs["alpha"], 

                # 'h'         : fff[key].attrs['h'],
                # 'omch2'     : fff[key].attrs['omch2'],
                # 'As1e9'     : fff[key].attrs['As1e9'],
                # 'ns'        : fff[key].attrs['ns'],
                # 'log10kh'   : fff[key]['log10kh'][...],
                # 'log10Pk'   : fff[key]['log10Pk'][...],
            }))
        df_all = pd.concat(_lst)
        df_all.to_csv(
            OUTFILEPATH + f'HOD_{flag}.csv',
            index=False,
        )
        
        fff.close()



read_hdf5_files()
# read_csv_original()
# read_HOD_fiducial_hdf5()