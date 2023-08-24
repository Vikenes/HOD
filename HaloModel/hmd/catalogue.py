from typing import List, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import scipy
import numpy as np
import pandas as pd
import hmd
from halotools.empirical_models import (
    halo_mass_to_halo_radius,
    halo_mass_to_virial_velocity,
)
from scipy.spatial import cKDTree


class Catalogue(ABC):
    def __len__(self,):
        return len(self.pos)

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        boxsize: float,
        cosmology: Optional = None,
        redshift: Optional = None,
    ):
        pos = df[["x", "y", "z"]].values
        vel = df[["v_x", "v_y", "v_z"]].values
        cols_1d = [
            c for c in df.columns if c not in ("x", "y", "z", "v_x", "v_y", "v_z")
        ]
        df_dict = {col: df[col].values for col in cols_1d}
        return cls(
            pos=pos,
            vel=vel,
            boxsize=boxsize,
            cosmology=cosmology,
            redshift=redshift,
            **df_dict
        )

    @classmethod
    def from_csv(
        cls,
        path: Path,
        boxsize: float,
        cosmology: Optional = None,
        redshift: Optional = None,
    ):
        df = pd.read_csv(path)
        return cls.from_frame(
            df=df, boxsize=boxsize, cosmology=cosmology, redshift=redshift,
        )

    def to_frame(self,):
        x, y, z = self.pos.T
        v_x, v_y, v_z = self.vel.T
        dict_attrs = {
            attr: getattr(self, attr)
            for attr in self.attrs_to_frame
            if getattr(self, attr) is not None
        }
        dict_attrs["x"] = x
        dict_attrs["y"] = y
        dict_attrs["z"] = z
        dict_attrs["v_x"] = v_x
        dict_attrs["v_y"] = v_y
        dict_attrs["v_z"] = v_z
        return pd.DataFrame(dict_attrs)

    def to_nbodykit(self,):
        from nbodykit.lab import ArrayCatalog

        return ArrayCatalog(
            {"Position": self.pos, "Velocity": self.vel,}, BoxSize=self.boxsize,
        )

    def cut_by_number_density(self, ndens: float):
        n_objects = int(ndens * self.boxsize ** 3)
        sorted_mass_idx = np.argsort(self.mass)
        pos = self.pos[sorted_mass_idx][-n_objects:, :]
        vel = self.vel[sorted_mass_idx, :][-n_objects:, :]
        mass = self.mass[sorted_mass_idx][-n_objects:]
        return pos, vel, mass

    def compute_tpcf(
        self,
        edges,
        ndens: Optional[float] = None,
        nthreads: int = 6,
        projected: bool = False,
    ):
        from pycorr import TwoPointCorrelationFunction

        if ndens is not None:
            pos, _, _ = self.cut_by_number_density(ndens)
        else:
            pos = self.pos
        return TwoPointCorrelationFunction(
            "theta" if projected else "s",
            edges,
            data_positions1=pos,
            engine="corrfunc",
            boxcenter=self.boxcenter,
            los="z",
            boxsize=self.boxsize,
            position_type="pos",
            nthreads=nthreads,
        )

    def compute_tpcf_s_mu(
        self, edges, ndens: Optional[float] = None, nthreads: int = 4,
    ):
        from pycorr import TwoPointCorrelationFunction

        if ndens is not None:
            pos, vel, _ = self.cut_by_number_density(ndens)
        else:
            pos = self.pos.copy()
            vel = self.vel.copy()
        z_pos = self.apply_redshift_distortion(pos[:, -1], vel[:, -1])
        pos[:, -1] = z_pos
        return TwoPointCorrelationFunction(
            "smu",
            edges,
            data_positions1=pos,
            engine="corrfunc",
            boxcenter=self.boxcenter,
            los="z",
            boxsize=self.boxsize,
            position_type="pos",
            nthreads=nthreads,
        )

    def apply_redshift_distortion(self, pos_z, vel_z):
        scale_factor = 1.0 / (1.0 + self.redshift)
        pos_err = vel_z / 100.0 / self.cosmology.efunc(self.redshift) / scale_factor
        zspace_pos = pos_z + pos_err
        if self.boxsize is not None:
            zspace_pos = zspace_pos % self.boxsize
        return zspace_pos

    def get_smoothed_density(self, smoothing_radius: float, nmesh: int):
        from pyrecon.mesh import RealMesh

        mesh = RealMesh(
            boxsize=self.boxsize,
            boxcenter=self.boxsize / 2.0,
            nmesh=nmesh,
            dtype="f8",
            nthreads=4,
        )
        mesh.assign_cic(self.pos, wrap=True)
        mesh_delta = mesh / np.mean(mesh) - 1.0
        mesh_delta.smooth_gaussian(radius=smoothing_radius, wrap=True)
        return mesh_delta.read_cic(positions=self.pos, wrap=True)

    def assign_ranks(self, smoothing_radius: float, nmesh: int, n_mass_bins: int = 50):
        mass_bins = np.logspace(
            np.log10(np.min(self.mass)), np.log10(np.max(self.mass)), n_mass_bins,
        )
        smoothed_delta = self.get_smoothed_density(
            smoothing_radius=smoothing_radius, nmesh=nmesh
        )
        idx = np.arange(len(self))
        ranks = np.zeros_like(self.mass)
        for i in range(len(mass_bins) - 1):
            mass_mask = (
                (self.mass >= mass_bins[i]) & (self.mass < mass_bins[i + 1])
            )
            mass_idx = idx[mass_mask]
            delta_mass = smoothed_delta[mass_mask]
            rank_mass = (scipy.stats.rankdata(delta_mass) - 1) / len(delta_mass)
            ranks[mass_idx] = rank_mass
        self.rank = ranks

    def compute_environment_overdensity(
        self, radius: float, remove_self: bool = False, n_threads: int = 1
    ):
        # Compute relative density of dark matter haos with top-hat window
        tree = cKDTree(self.pos, boxsize=self.boxsize,)
        neighbours = tree.query_ball_point(self.pos, r=radius, n_jobs=n_threads,)
        env_mass = np.array([sum(self.mass[idx]) for idx in neighbours])
        if remove_self:
            env_mass -= self.mass
        mean_density = len(self) / self.boxsize ** 3
        return env_mass / mean_density


class ParticleCatalogue(Catalogue):
    def __init__(
        self,
        pos: np.array,
        vel: np.array,
        boxsize: float,
        n_particles_by_halo: np.array = None,
        build_tree: bool = False,
    ):
        self.pos = pos
        self.vel = vel
        self.boxsize = boxsize
        if build_tree:
            self.tree = cKDTree(self.pos, boxsize=self.boxsize,)
        else:
            self.tree = None
        self.n_particles_by_halo = n_particles_by_halo
        if self.n_particles_by_halo is not None:
            self.start_id_per_halo = (
                np.cumsum(self.n_particles_by_halo) - self.n_particles_by_halo
            )

    def get_pos_vel_in_halo(self, halo_idx: int) -> np.array:
        return (
            self.pos[
                self.start_id_per_halo[halo_idx] : self.start_id_per_halo[halo_idx]
                + self.n_particles_by_halo[halo_idx]
            ],
            self.vel[
                self.start_id_per_halo[halo_idx] : self.start_id_per_halo[halo_idx]
                + self.n_particles_by_halo[halo_idx]
            ],
        )


class HaloCatalogue(Catalogue):
    def __init__(
        self,
        pos: np.array,
        vel: np.array,
        mass: np.array,
        boxsize: float,
        rank: Optional[np.array] = None,
        radius: Optional[np.array] = None,
        vel_disp: Optional[np.array] = None,
        concentration: Optional[np.array] = None,
        mdef: str = "200m",
        cosmology: Optional = None,
        redshift: Optional[float] = None,
        conc_mass_model: Optional = None,
        get_delta_environment: bool = False,
        original_idx: Optional[np.array] = None,
    ):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.rank = rank
        self.boxsize = boxsize
        self.boxcenter = self.boxsize / 2.0
        self.redshift = redshift
        self.cosmology = cosmology
        if radius is not None:
            self.radius = radius
        else:
            self.radius = halo_mass_to_halo_radius(
                mass=mass, cosmology=cosmology, redshift=redshift, mdef=mdef,
            ) * (1.0 + redshift)
        if vel_disp is not None:
            self.vel_disp = vel_disp
        else:
            self.vel_disp = (
                halo_mass_to_virial_velocity(
                    mass, cosmology=cosmology, redshift=redshift, mdef=mdef
                )
                / np.sqrt(2)
                / np.sqrt(1 + redshift)
            )/np.sqrt(3)
        if concentration is not None:
            self.concentration = concentration
        elif conc_mass_model is not None:
            self.concentration = conc_mass_model(
                prim_haloprop=mass,
                cosmology=cosmology,
                redshift=redshift,
                sigma8=cosmology.get_sigma8()
                if hasattr(cosmology, "get_sigma8")
                else None,
            )
        else:
            concentration = None
        if cosmology is not None:
            z_pos = self.apply_redshift_distortion(self.pos[:, -1], self.vel[:, -1])
            self.s_pos = self.pos.copy()
            self.s_pos[:, -1] = z_pos
        if get_delta_environment:
            self.delta = self.compute_environment_overdensity(radius=10.0, n_threads=16)
        else:
            self.delta = None
        self.original_idx = original_idx
        self.attrs_to_frame = ["mass", "radius", "vel_disp", "concentration","rank"]

    @classmethod
    def from_dq(
        cls,
        run: int,
        snapshot: int,
        resolution: Optional[str] = "LR",
        log_M_min: Optional[float] = None,
        log_M_max: Optional[float] = None,
        mass_correction: bool = False,
        LR_run =None,
    ):
        from dq import read_halo_data, snapshot_to_redshift, Cosmology, constants, convert_run_to_cosmo_number
        from hmd.concentration import diemer15

        pos, vel, mass = read_halo_data(
            run=run, snapshot=snapshot, high_res=True if resolution == "HR" else False
        )
        pos = pos[mass > 0.]
        vel = vel[mass > 0.]
        mass = mass[mass > 0.]
        if mass_correction:
            mass_list = np.load('/cosma7/data/dp004/dc-cues1/DarkQuest/halo_data/mass_list_LR.npy')
            if resolution == 'LR' or LR_run > 100:
                particle_mass = mass_list[0]
            else:
                cnumber = convert_run_to_cosmo_number(LR_run)
                particle_mass = mass_list[cnumber]
            if resolution == "HR":
                particle_mass /= 8.
            n_particles = mass // particle_mass
            correction_factor = 1.0 + n_particles ** (-0.55)
            mass *= correction_factor
        if log_M_min is not None or log_M_max is not None:
            mass_mask = (mass >= 10 ** log_M_min) & (mass <= 10 ** log_M_max)
            pos = pos[mass_mask]
            vel = vel[mass_mask]
            mass = mass[mass_mask]
        cosmology = Cosmology.from_run(run=run)
        return cls(
            pos=pos,
            vel=vel,
            mass=mass,
            mdef="200m",
            cosmology=cosmology,
            redshift=snapshot_to_redshift(snapshot),
            boxsize=2000.0 if resolution == "LR" else 1000.0,
            conc_mass_model=hmd.concentration.diemer15,
        )

    @property
    def n_halos(self,):
        return len(self.pos)

    def get_particles_in_halos(self, particle_catalogue, n_threads: int = -1):
        particles_in_halo = particle_catalogue.tree.query_ball_point(
            self.pos, self.radius, workers=n_threads
        )
        n_in_halo = [len(p_in_halo) for p_in_halo in particles_in_halo]
        all_particle_ids = [item for sublist in particles_in_halo for item in sublist]
        positions = particle_catalogue.pos[all_particle_ids]
        velocities = particle_catalogue.vel[all_particle_ids]
        return positions, velocities, n_in_halo

    def compute_hmf(self, edges, differential=True):
        num_halos = np.histogram(self.mass, edges)[0]
        if differential:
            M_width = edges[1:] - edges[:-1]
            return num_halos / self.boxsize ** 3 / M_width
        else:
            num_halos = np.cumsum(num_halos[::-1])[::-1]
            return num_halos / self.boxsize ** 3


class GalaxyCatalogue(Catalogue):
    def __init__(
        self,
        pos: np.array,
        vel: np.array,
        host_mass: np.array,
        host_radius: np.array,
        host_centric_distance: np.array,
        host_id: np.array,
        galaxy_type: np.array,
        boxsize: float,
        host_rank: Optional[np.array] = None,
        host_concentration: Optional[np.array] = None,
        redshift: Optional[float] = None,
        cosmology: Optional = None,
    ):
        self.pos = pos
        self.vel = vel
        self.boxsize = boxsize
        self.boxcenter = self.boxsize / 2.0
        self.host_mass = host_mass
        self.host_radius = host_radius
        self.host_rank = host_rank
        self.host_concentration = host_concentration
        self.host_centric_distance = host_centric_distance
        self.host_id = host_id
        self.galaxy_type = galaxy_type
        self.attrs_to_frame = [
            "host_mass",
            "host_radius",
            "host_concentration",
            "host_rank",
            "host_centric_distance",
            "host_id",
            "galaxy_type",
        ]
        self.cosmology = cosmology
        self.redshift = redshift
        if cosmology is not None:
            z_pos = self.apply_redshift_distortion(self.pos[:, -1], self.vel[:, -1])
            self.s_pos = self.pos.copy()
            self.s_pos[:, -1] = z_pos

    @property
    def n_galaxies(self,):
        return len(self.pos)

    @property
    def n_satellites(self,):
        return len(self.galaxy_type[self.galaxy_type == "satellite"])

    @property
    def n_centrals(self,):
        return len(self.galaxy_type[self.galaxy_type == "central"])

    @property
    def f_satellites(self,):
        return self.n_satellites / self.n_galaxies

    @property
    def n_density_galaxies(self,):
        return len(self.pos) / self.boxsize ** 3
