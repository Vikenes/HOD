import numpy as np
import pandas as pd
import logging
import time
from typing import Optional

from hmd.catalogue import ParticleCatalogue, HaloCatalogue, GalaxyCatalogue
from hmd.occupation import Occupation
from hmd.profiles import FixedCosmologyProfile
from hmd.galaxy import Galaxy


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class HODMaker:
    def __init__(
        self,
        halo_catalogue: HaloCatalogue,
        central_occ: Occupation,
        sat_occ: Occupation,
        satellite_profile: FixedCosmologyProfile,
        galaxy: Galaxy,
        particle_catalogue: Optional[ParticleCatalogue] = None,
    ):
        self.halo_catalogue = halo_catalogue
        self.boxsize = self.halo_catalogue.boxsize
        self.halo_df = halo_catalogue.to_frame()
        self.central_occ = central_occ
        self.sat_occ = sat_occ
        self.satellite_profile = satellite_profile
        self.galaxy = galaxy
        self.particle_catalogue = particle_catalogue

    def get_central_df(self,):
        probability = self.central_occ.get_n(
            halo_mass=self.halo_catalogue.mass,
            halo_rank=self.halo_catalogue.rank,
            galaxy=self.galaxy,
        )
        randoms = np.random.random(self.halo_catalogue.n_halos)
        central_df = self.halo_df.loc[self.halo_df.index[probability > randoms]]
        v_disp = self.central_occ.get_velocity_dispersion(
            central_df["vel_disp"].values, galaxy=self.galaxy
        )
        central_df["v_x"] += np.random.normal(loc=np.zeros_like(v_disp), scale=v_disp)
        central_df["v_y"] += np.random.normal(loc=np.zeros_like(v_disp), scale=v_disp)
        central_df["v_z"] += np.random.normal(loc=np.zeros_like(v_disp), scale=v_disp)
        return central_df

    def get_satellite_df(
        self, centrals_df: pd.DataFrame,
    ):
        if 'rank' in centrals_df.columns:
            rank =  centrals_df['rank'].values
        else:
            rank = None
        lambda_sat = self.sat_occ.lambda_sat(
            centrals_df["mass"].values,
            halo_rank=rank,
            galaxy=self.galaxy,
        )
        n_sats = np.random.poisson(lam=lambda_sat)
        return centrals_df.loc[centrals_df.index.repeat(n_sats)]

    def sample_positions_from_profile(
        self, halo_concentration: np.array, n_points: int, **kwargs
    ):
        # Assumption: Isotropic profile
        halo_concentration = self.sat_occ.get_concentration(
            halo_concentrations=halo_concentration, galaxy=self.galaxy,
        )
        r = self.satellite_profile.mc_generate_radial_positions(
            halo_concentration=halo_concentration,
            num_pts=n_points,
            halo_radius=1.0,
            **kwargs,
        )
        cos_t = np.random.uniform(-1.0, 1.0, n_points)
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        sin_t = np.sqrt((1.0 - cos_t * cos_t))
        x = r * sin_t * np.cos(phi)
        y = r * sin_t * np.sin(phi)
        z = r * cos_t
        return r, x, y, z

    def sample_velocities_from_profile(self, halo_vel_disp: np.array):
        # TODO: carefule with random seed, you don't want ot have
        # the same on all velocity components!
        satellite_vel_disp = self.sat_occ.get_velocity_dispersion(
            halo_dispersions=halo_vel_disp, galaxy=self.galaxy,
        )
        v_x = self.satellite_profile.mc_generate_radial_velocities(
            velocity_dispersion=satellite_vel_disp,
        )
        v_y = self.satellite_profile.mc_generate_radial_velocities(
            velocity_dispersion=satellite_vel_disp,
        )
        v_z = self.satellite_profile.mc_generate_radial_velocities(
            velocity_dispersion=satellite_vel_disp,
        )
        return v_x, v_y, v_z

    def add_profile(self, satellite_df: pd.DataFrame):
        # TODO: Computing concentrations super slow with diemer15
        (
            halo_centric_r,
            halo_centric_x,
            halo_centric_y,
            halo_centric_z,
        ) = self.sample_positions_from_profile(
            halo_concentration=satellite_df["concentration"],
            n_points=len(satellite_df),
        )
        satellite_df["host_centric_distance"] = 0.0
        satellite_df.loc[:, "host_centric_distance"] = (
            satellite_df.loc[:, "radius"] * halo_centric_r
        )
        satellite_df.loc[:, "x"] += satellite_df.loc[:, "radius"] * halo_centric_x
        satellite_df.loc[:, "y"] += satellite_df.loc[:, "radius"] * halo_centric_y
        satellite_df.loc[:, "z"] += satellite_df.loc[:, "radius"] * halo_centric_z
        v_x, v_y, v_z = self.sample_velocities_from_profile(
            halo_vel_disp=satellite_df["vel_disp"]
        )
        satellite_df.loc[:, "v_x"] += v_x
        satellite_df.loc[:, "v_y"] += v_y
        satellite_df.loc[:, "v_z"] += v_z
        return satellite_df

    def sample_particles(self, satellite_df: pd.DataFrame):
        for (halo_id, group_satellites) in satellite_df.groupby("host_id"):
            particle_pos, particle_vel = self.particle_catalogue.get_pos_vel_in_halo(
                halo_idx=halo_id
            )
            particle_idx = np.random.choice(
                range(len(particle_pos)), size=len(group_satellites)
            )
            satellite_df.loc[halo_id, "x"] = particle_pos[particle_idx, 0]
            satellite_df.loc[halo_id, "y"] = particle_pos[particle_idx, 1]
            satellite_df.loc[halo_id, "z"] = particle_pos[particle_idx, 2]
            satellite_df.loc[halo_id, "v_x"] = particle_vel[particle_idx, 0]
            satellite_df.loc[halo_id, "v_y"] = particle_vel[particle_idx, 1]
            satellite_df.loc[halo_id, "v_z"] = particle_vel[particle_idx, 2]
            # Add satellite velocity bias
        return satellite_df

    def enforce_periodic_box(self, pos):
        return pos % self.boxsize

    def __call__(self, sample_particles=False):
        logger.info(f"Sampling HOD at z = {self.halo_catalogue.redshift}")
        logger.info("Sampling centrals ...")
        central_df = self.get_central_df()
        central_df["galaxy_type"] = "central"
        central_df["host_centric_distance"] = 0.0
        central_df["host_id"] = central_df.index
        if self.sat_occ is not None:
            logger.info("Sampling satellites ...")
            satellite_df = self.get_satellite_df(central_df,)
            satellite_df["galaxy_type"] = "satellite"
            satellite_df["host_id"] = satellite_df.index
            if sample_particles:
                satellite_df = self.sample_particles(satellite_df=satellite_df)
            else:
                if self.satellite_profile is not None:
                    logger.info("Moving satellites around to fit profile ...")
                    satellite_df = self.add_profile(satellite_df=satellite_df)
                else:
                    satellite_df["host_centric_distance"] = 0.0
            logger.info("Done centrals and satellites")
            galaxy_df = pd.concat([central_df, satellite_df], ignore_index=True,)
        else:
            galaxy_df = central_df
        columns_to_rename = [
            "mass",
            "radius",
            "concentration",
            "rank",
        ]
        column_mapper = {c: f"host_{c}" for c in columns_to_rename}
        galaxy_df.rename(columns=column_mapper, inplace=True)
        galaxy_df = galaxy_df.drop(columns=["vel_disp"])
        galaxy_df["x"] = self.enforce_periodic_box(galaxy_df["x"])
        galaxy_df["y"] = self.enforce_periodic_box(galaxy_df["y"])
        galaxy_df["z"] = self.enforce_periodic_box(galaxy_df["z"])
        # apply PBC
        self.galaxy_cat = GalaxyCatalogue.from_frame(
            galaxy_df,
            boxsize=self.halo_catalogue.boxsize,
            cosmology=self.halo_catalogue.cosmology,
            redshift=self.halo_catalogue.redshift,
        )
        self.galaxy_df = galaxy_df
