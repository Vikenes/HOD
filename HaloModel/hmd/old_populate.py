import numpy as np
import pandas as pd
from pathlib import Path
from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
from halotools.empirical_models import (
    NFWPhaseSpace,
    Zheng07Sats,
    halo_mass_to_halo_radius,
)
from hmd.profiles import SimpleProfile, ConstantProfile
from halotools.mock_observables import tpcf_multipole
from dq import snapshot_to_redshift, read_halo_data
from dq import constants
from dq.cosmology import Cosmology
from gsm.measurements.tpcf import compute_real_tpcf, compute_tpcf_s_mu
import argparse

# RHO_CR = 3.6149635e11
# RHO_M = 0.4745560551231544 * RHO_CR
OUTPUT_DATA_DIR = Path("/cosma7/data/dp004/dc-cues1/DarkQuest/mock_catalogues/")


def sample_positions(profile, n_points):
    r = profile.mc_generate_radial_positions(num_pts=n_points, halo_radius=1.0)
    cos_t = np.random.uniform(-1.0, 1.0, n_points)
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    sin_t = np.sqrt((1.0 - cos_t * cos_t))
    x = r * sin_t * np.cos(phi)
    y = r * sin_t * np.sin(phi)
    z = r * cos_t
    return r, x, y, z


def add_profile(profile, galaxy_df):
    modified_galaxy_df = galaxy_df.copy()
    satellite_mask = galaxy_df["gal_type"] == "satellites"
    halo_centric_r, halo_centric_x, halo_centric_y, halo_centric_z = sample_positions(
        profile, len(galaxy_df[satellite_mask])
    )
    modified_galaxy_df["halo_centric_distance"] = 0.0
    modified_galaxy_df.loc[satellite_mask, "halo_centric_distance"] = (
        galaxy_df.loc[satellite_mask, "halo_r200m"] * halo_centric_r
    )
    modified_galaxy_df.loc[satellite_mask, "x"] += (
        galaxy_df.loc[satellite_mask, "halo_r200m"] * halo_centric_x
    )
    modified_galaxy_df.loc[satellite_mask, "y"] += (
        galaxy_df.loc[satellite_mask, "halo_r200m"] * halo_centric_y
    )
    modified_galaxy_df.loc[satellite_mask, "z"] += (
        galaxy_df.loc[satellite_mask, "halo_r200m"] * halo_centric_z
    )
    return modified_galaxy_df


def get_halo_table(run: int, snapshot: int, boxsize: float, cosmology):
    pos, vel, m200c = read_halo_data(run=run, snapshot=snapshot)
    """
    mass_cut = (m200c > pow(10,13.)) & (m200c < pow(10,14.5))
    pos = pos[mass_cut]
    vel = vel[mass_cut]
    m200c = m200c[mass_cut]
    """
    redshift = snapshot_to_redshift(snapshot=snapshot)
    halo_ids = np.arange(len(pos))
    return UserSuppliedHaloCatalog(
        redshift=redshift,
        Lbox=boxsize,
        particle_mass=1.0e2,
        halo_x=pos[:, 0],
        halo_y=pos[:, 1],
        halo_z=pos[:, 2],
        halo_vx=vel[:, 0],
        halo_vy=vel[:, 1],
        halo_vz=vel[:, 2],
        halo_id=halo_ids,
        halo_mvir=m200c,
        halo_m200m=m200c,
        halo_upid=np.array([-1] * len(pos)),
        halo_hostid=halo_ids,
        halo_r200m=halo_mass_to_halo_radius(m200c, cosmology, redshift, mdef="200m"),
        halo_rvir=halo_mass_to_halo_radius(m200c, cosmology, redshift, mdef="200m"),
    )


def get_galaxy_table(
    halo_table,
    redshift,
    cosmology,
    logMmin=13.3,
    sigma_logM=0.26,
    kappa=0.5,
    logM1=13.8,
    alpha=0.3,
    boxsize=2000.0,
    profile: str = "nwf",
):
    cens_occ_model = Zheng07Cens(redshift=redshift, prim_haloprop_key="halo_m200m")
    cens_occ_model.param_dict["logMmin"] = logMmin
    cens_occ_model.param_dict["sigma_logM"] = sigma_logM
    cens_prof_model = TrivialPhaseSpace(redshift=redshift, mdef="200m")

    sats_occ_model = Zheng07Sats(
        redshift=redshift,
        modulate_with_cenocc=True,
        cenocc_model=cens_occ_model,
        prim_haloprop_key="halo_m200m",
    )
    sats_occ_model.param_dict["logM0"] = np.log10(
        kappa * 10 ** cens_occ_model.param_dict["logMmin"]
    )
    sats_occ_model.param_dict["logM1"] = logM1
    sats_occ_model.param_dict["alpha"] = alpha
    if profile == "nfw":
        sats_prof_model = NFWPhaseSpace(
            redshift=redshift,
            cosmology=cosmology,
            mdef="200m",
            halo_boundary_key="halo_r200m",
            conc_mass_model="dutton_maccio14",
        )
    else:
        sats_prof_model = TrivialPhaseSpace(redshift=redshift, mdef="200m")

    model_instance = HodModelFactory(
        centrals_occupation=cens_occ_model,
        centrals_profile=cens_prof_model,
        satellites_occupation=sats_occ_model,
        satellites_profile=sats_prof_model,
        redshift=redshift,
    )
    model_instance.populate_mock(halo_table)
    return model_instance.mock.galaxy_table


def filter_table(table, galaxy_type):
    mask = table["gal_type"] == galaxy_type
    return table[mask]


def get_positions_from_table(table):
    pos = np.stack(
        [np.array(table["x"]), np.array(table["y"]), np.array(table["z"]),]
    ).T
    vel = np.stack(
        [np.array(table["vx"]), np.array(table["vy"]), np.array(table["vz"]),]
    ).T
    return pos, vel


def compute_xi(r, pos, boxsize, pos_cross=None):
    r_c = 0.5 * (r[1:] + r[:-1])
    tpcf = compute_real_tpcf(
        r=r, pos=pos, pos_cross=pos_cross, boxsize=boxsize, num_threads="max"
    )
    return pd.DataFrame({"r_c": r_c, "xi": tpcf})


def compute_multipoles(
    r, pos, vel, cosmology, boxsize, redshift, los=0, pos_cross=None, vel_cross=None
):
    r_c = 0.5 * (r[1:] + r[:-1])
    mu = (1 + np.geomspace(-0.001, -1, 60))[::-1]
    s_tpcf = compute_tpcf_s_mu(
        s=r,
        mu=mu,
        pos=pos,
        vel=vel,
        los_direction=los,
        cosmology=cosmology,
        boxsize=boxsize,
        redshift=redshift,
        num_threads="max",
        pos_cross=pos_cross,
        vel_cross=vel_cross,
    )
    monopole = tpcf_multipole(s_tpcf, mu, order=0)
    quadrupole = tpcf_multipole(s_tpcf, mu, order=2)
    hexadecapole = tpcf_multipole(s_tpcf, mu, order=4)
    return pd.DataFrame(
        {
            "s_c": r_c,
            "monopole": monopole,
            "quadrupole": quadrupole,
            "hexadecapole": hexadecapole,
        }
    )


if __name__ == "__main__":
    profile = "constant"
    planck_runs = np.arange(101, 116)
    r = np.logspace(-2, np.log10(150.0), 150)
    s = np.logspace(-2, np.log10(60.0), 70)
    snapshot = 15
    redshift = snapshot_to_redshift(snapshot)
    cosmology = Cosmology.from_run(run=101)
    for run in planck_runs:
        print(f"Run = {run}")
        halo_table = get_halo_table(
            run=run, snapshot=snapshot, boxsize=constants.boxsize, cosmology=cosmology
        )
        galaxy_table = get_galaxy_table(
            halo_table, redshift=redshift, cosmology=cosmology, profile=profile
        )
        galaxy_df = galaxy_table.to_pandas()
        if profile != "nfw":
            sp = ConstantProfile(cosmology=cosmology, redshift=redshift)
            galaxy_df = add_profile(profile=sp, galaxy_df=galaxy_df)

        galaxy_types = ["s_s", "c_c", "g_g", "c_s"]
        for gal_type in galaxy_types:
            if gal_type == "c_s":
                pos_left, vel_left = get_positions_from_table(
                    filter_table(galaxy_df, "centrals")
                )
                pos_right, vel_right = get_positions_from_table(
                    filter_table(galaxy_df, "satellites")
                )
                xi = compute_xi(
                    r, pos_left, boxsize=constants.boxsize, pos_cross=pos_right
                )
                xi_multipoles = compute_multipoles(
                    r=s,
                    pos=pos_left,
                    vel=vel_left,
                    cosmology=cosmology,
                    boxsize=constants.boxsize,
                    redshift=redshift,
                    pos_cross=pos_right,
                    vel_cross=vel_right,
                )

            elif gal_type in ("c_c", "s_s"):
                if gal_type == "c_c":
                    pos, vel = get_positions_from_table(
                        filter_table(galaxy_df, "centrals")
                    )
                elif gal_type == "s_s":
                    pos, vel = get_positions_from_table(
                        filter_table(galaxy_df, "satellites")
                    )
                xi = compute_xi(r, pos, boxsize=constants.boxsize)
                xi_multipoles = compute_multipoles(
                    r=s,
                    pos=pos,
                    vel=vel,
                    cosmology=cosmology,
                    boxsize=constants.boxsize,
                    redshift=redshift,
                )

            else:
                pos, vel = get_positions_from_table(galaxy_df)
                xi = compute_xi(r, pos, boxsize=constants.boxsize)
                xi_multipoles = compute_multipoles(
                    r=s,
                    pos=pos,
                    vel=vel,
                    cosmology=cosmology,
                    boxsize=constants.boxsize,
                    redshift=redshift,
                )
            xi.to_csv(
                OUTPUT_DATA_DIR
                / f"summary_statistics/my_xi_real/constant_{gal_type}_xi_run{str(run).zfill(3)}_s{str(snapshot).zfill(3)}.csv",
                index=False,
            )
            xi_multipoles.to_csv(
                OUTPUT_DATA_DIR
                / f"summary_statistics/my_xi_s/constant_{gal_type}_xi_l_run{str(run).zfill(3)}_s{str(snapshot).zfill(3)}_los0.csv",
                index=False,
            )
            galaxy_df = galaxy_df[
                [
                    "x",
                    "y",
                    "z",
                    "vx",
                    "vy",
                    "vz",
                    "gal_type",
                    "halo_hostid",
                    "halo_x",
                    "halo_y",
                    "halo_z",
                    "halo_mvir",
                    "halo_rvir",
                ]
            ]
            galaxy_df.to_csv(
                OUTPUT_DATA_DIR / f"my_mocks/constant_R{run}_S{snapshot}.csv",
                index=False,
            )
