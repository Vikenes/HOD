import numpy as np
import sys
import h5py
import pandas as pd  

import dq
from hmdMG.catalogue import HaloCatalogue

from cls_measure_xiS import measure_xiS_s_mu



dataflag = 'DQx14'

box_size = 2000.0

snap_num = 15
redshift = dq.snapshots.snapshot_to_redshift(snapshot=snap_num,)
print(f"redshift = {redshift:.2f}")

# r_bin_edge = np.arange(0, 131, 1)
s_bin_edge = np.logspace(np.log10(0.7), np.log10(130), 260)
mu_bin_edge = np.linspace(-1, 1, 1024)

_ng_lst = []
_nc_lst = []
_ns_lst = []
for box_index in range(101, 114+1):
    df = pd.read_hdf(
        f"./data/HOD_{dataflag}_redshift{redshift:.2f}_box{box_index}.hdf",
        key='HOD', 
    )

    Ng = len(df)
    ng = Ng / (box_size**3)
    _ng_lst.append(ng)

    dfc = df[
        df['galaxy_type'] == 'central'
    ]
    Nc = len(dfc)
    nc = Nc / (box_size**3)
    _nc_lst.append(nc)


    dfs = df[
        df['galaxy_type'] == 'satellite'
    ]
    Ns = len(dfs)
    ns = Ns / (box_size**3)
    _ns_lst.append(ns)


    print(f'box{box_index} done')

ng = np.mean(np.array(_ng_lst))
nc = np.mean(np.array(_nc_lst))
ns = np.mean(np.array(_ns_lst))

np.savetxt(
    f'./data/numden_g_c_s_{dataflag}_redshift{redshift:.2f}.dat',
    np.array([ng, nc, ns]).T,
)
