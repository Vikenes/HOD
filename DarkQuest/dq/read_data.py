import numpy as np
from pathlib import Path
import pandas as pd

from dq import snapshot_to_scale_factor

HIGHRES_DATA_DIR = Path("/cosma7/data/dp004/dc-cues1/DarkQuest/high_res/DQ1_HR/")
LOWRES_DATA_DIR = Path("/cosma6/data/dp004/dc-cues1/DarkQuest/baorsd/")
HOD_DATA_DIR = Path("/cosma7/data/dp004/dc-cues1/DarkQuest/mock_catalogues/")


def read_hod_data(
    run: int, 
    snapshot: int, 
    galaxy_type: str = "all",
):
    return read_custom_hod(
        run=run, 
        snapshot=snapshot, 
        galaxy_type=galaxy_type,
    )



def read_custom_hod(
    run: int, 
    snapshot: int, 
    galaxy_type: str,
):
    df = pd.read_csv(HOD_DATA_DIR / f"my_mocks/R{run}_S{snapshot}.csv")

    if galaxy_type != "all":
        df = df[df["gal_type"] == galaxy_type]
    pos = np.stack([np.array(df["x"]), np.array(df["y"]), np.array(df["z"]),]).T
    vel = np.stack([np.array(df["vx"]), np.array(df["vy"]), np.array(df["vz"]),]).T
    return pos, vel



def read_yosuke_hod(
    run: int, 
    snapshot: int,
):
    filename_root = f"fiducial/R{str(run).zfill(3)}_S{str(snapshot).zfill(3)}_rockstar_gal_2Gpc_Mcorr_nfwcm_wFoG"
    filename_end = "bin"
    pos = np.fromfile(
        HOD_DATA_DIR / (filename_root + f"_pos.{filename_end}"), dtype=np.float32
    )
    pos = np.reshape(pos, (-1, 3))
    vel = np.fromfile(
        HOD_DATA_DIR / (filename_root + f"_vel.{filename_end}"), dtype=np.float32
    )
    vel = np.reshape(vel, (-1, 3))
    vel = correct_velocities_gadget(vel=vel, snapshot=snapshot)
    return pos, vel



def read_halo_data(
    run: int, 
    snapshot: int, 
    high_res=False,
):
    if high_res:
        return read_high_res_halo_data(
            run=run, 
            snapshot=snapshot,
        )
    return read_low_res_halo_data(
        run=run, 
        snapshot=snapshot,
    )



def correct_velocities_gadget(
    vel, 
    snapshot,
):
    return vel * np.sqrt(snapshot_to_scale_factor(snapshot=snapshot))



def read_high_res_halo_data(
    run: int, 
    snapshot: int,
):
    filename_root = f"R{str(run).zfill(3)}_S{str(snapshot).zfill(3)}_rockstar_central"
    filename_end = "npy"
    pos = np.load(HIGHRES_DATA_DIR / (filename_root + f"_pos.{filename_end}"))
    vel = np.load(HIGHRES_DATA_DIR / (filename_root + f"_vel.{filename_end}"))
    vel = correct_velocities_gadget(vel=vel, snapshot=snapshot)
    m200c = np.load(HIGHRES_DATA_DIR / (filename_root + f"_mass.{filename_end}"))
    return pos, vel, m200c



def read_low_res_halo_data(
    run: int, 
    snapshot: int,
):
    DATA_DIR = LOWRES_DATA_DIR / f"run{str(run).zfill(3)}/halo_catalog/"
    if run <= 60 or (run >= 101 and run != 115 and run < 1000):
        filename_root = f"S{str(snapshot).zfill(3)}_cen_rockstar"
    else:
        filename_root = f"R{str(run).zfill(3)}_S{str(snapshot).zfill(3)}"
    filename_end = "bin"
    pos = np.fromfile(
        DATA_DIR / (filename_root + f"_pos.{filename_end}"), dtype=np.float32
    )
    if (run == 53) & (snapshot == 18):
        pos = pos[:-349]
    vel = np.fromfile(
        DATA_DIR / (filename_root + f"_vel.{filename_end}"), dtype=np.float32
    )
    m200c = np.fromfile(
        DATA_DIR / (filename_root + f"_mass.{filename_end}"), dtype=np.float32
    )
    pos = np.reshape(pos, (-1, 3))
    vel = np.reshape(vel, (-1, 3))
    vel = correct_velocities_gadget(vel=vel, snapshot=snapshot)
    return pos, vel, m200c


def cut_by_number_density(
    pos: np.array,
    vel: np.array,
    m200c: np.array,
    number_density: float = 1.0e-3,
    boxsize: float = 2000.0,
):
    n_objects = int(number_density * boxsize ** 3)
    sorted_mass_idx = np.argsort(m200c)
    pos = pos[sorted_mass_idx][-n_objects:, :]
    vel = vel[sorted_mass_idx, :][-n_objects:, :]
    m200c = m200c[sorted_mass_idx][-n_objects:]
    return pos, vel, m200c
