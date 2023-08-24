from abc import ABC, abstractmethod
from typing import Optional
from scipy import integrate

# from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import InterpolatedUnivariateSpline as ius

import numpy as np
from pathlib import Path
import pandas as pd
from dark_emulator.darkemu.hmf import hmf_gp
from hmd.cosmology import Cosmology
from hmd.utils import simps
from functools import partial
from jax import jit, vmap
import jax.numpy as jnp

default_hmf_path = Path("/cosma7/data/dp004/dc-cues1/DarkQuest/hmfs/")


class HaloMassFunction(ABC):
    @abstractmethod
    def get_dndM(self, halo_mass: np.array, **kwargs) -> np.array:
        """
        Get the cumulative halo number density at given
        halo mass thresholds

        Args:
            halo_mass (np.array): mass of the haloes

        Returns:
            np.array: number density of haloes above threshold
        """
        pass

    # @partial(jit, static_argnums=(0,))
    def convert_mass_to_density(
        self,
        mass: float,
        cosmology: Optional[Cosmology] = None,
        redshift: Optional[float] = None,
        upper_limit: float = pow(10, 15.95),
    ):
        """Convert halo mass into halo number density threshold,
        The minimum value this function can return is 1.e-17

        Args:
            mass (float): mass
            upper_limit (float): maximum halo mass to consider
        """
        mass_array = jnp.linspace(mass, upper_limit, 2001)
        dx = mass_array[1] - mass_array[0]
        y = jnp.nan_to_num(
            self.get_dndM(mass_array, cosmology=cosmology, redshift=redshift,)
        )
        return simps(y=y, n=len(y), dx=dx)

    @partial(jit, static_argnums=(0,))
    def vectorized_mass_to_density(
        self,
        mass: np.array,
        cosmology: Optional[Cosmology] = None,
        redshift: Optional[float] = None,
        upper_limit: float = 1.0e16,
    ):
        return vmap(
            self.convert_mass_to_density, in_axes=(0, None, None, None), out_axes=0
        )(mass, cosmology, redshift, upper_limit)

    def _compute_density_to_mass_table(
        self,
        cosmology: Optional[Cosmology] = None,
        redshift: Optional[float] = None,
        nint: int = 50,
    ):
        """Invert mass to density relation to find density to mass relation

        Args:
            nint (int): number of points used in interpolation
        """
        log_mass = jnp.linspace(12.0, 15.95, nint)
        densities = jnp.log(
            self.vectorized_mass_to_density(
                mass=10 ** log_mass, cosmology=cosmology, redshift=redshift,
            )
        )
        log_mass = log_mass[(~jnp.isnan(densities)) & (~jnp.isinf(densities))]
        sorted_idx = jnp.argsort(densities)[::-1]
        self.density_to_mass = ius(-densities[sorted_idx], log_mass[sorted_idx])

    @partial(jit, static_argnums=(0, 1))
    def convert_density_to_mass(
        self, density: float,
    ):
        """Convert halo number density into minimum halo mass
        (found by interpolating the inverse relation)

        Args:
            density (float): number density
        """
        return 10 ** self.density_to_mass(-np.log(density))


class MeasuredHaloMassFunction(HaloMassFunction):
    def __init__(
        self, mass: np.array, dndM: np.array,
    ):
        """Class to wrap a measured halo mass function
        and interpolate over it

        Args:
            mass (np.array): mass of the haloes
            dndM (np.array): dndM, measured halo mass function
            on mass bins
        """
        self.dndM = ius(mass, dndM, ext=1)

    # TODO: Should we use corrected masses and HR??
    @classmethod
    def from_path(
        cls,
        snapshot: int,
        hmf_path=default_hmf_path,
        resolution: str = "LR",
        mass_correction: bool = True,
    ):
        """Generate a halo mass function from a path to its
        mesurement

        Args:
            hmf_path: path to where the halo mass function is saved
            snapshot (int): snapshot of the simulation
            resolution (str): resolution (either HR or LR)
        """
        filename = f"S{str(snapshot).zfill(3)}_halocount_{resolution}"
        if mass_correction:
            filename += "_corrected"
        filename += ".dat"

        df = pd.read_csv(
            hmf_path / filename,
            skiprows=1,
            sep=" ",
            names=["M_min", "M_max", "N_halo", "error"],
        )
        df["Delta_M"] = df["M_max"] - df["M_min"]
        df["M"] = 0.5 * (df["M_min"] + df["M_max"])
        return cls(mass=df["M"].values, dndM=(df["N_halo"] / df["Delta_M"]).values)

    @classmethod
    def from_dq_high_res(
        cls, run: int, snapshot: int,
    ):
        DATA_DIR = Path(f"/cosma7/data/dp004/dc-cues1/DarkQuest/halo_data")
        particle_masses = (
            np.load(DATA_DIR / "mass_list_LR.npy")[run if run <= 100 else -1] / 8.0
        )
        if run > 100:
            n_count = np.load(DATA_DIR / "halocount_fiducial.npy")[0, snapshot]
        else:
            n_count = np.load(DATA_DIR / "halocount_varied.npy")[run, snapshot]
        M_bin = np.logspace(12, 16, 81)
        volume = 2000.0 ** 3 / 8  # Factor of 8 due to highres
        N_par = M_bin / particle_masses
        M_bin_corr = (1 + N_par ** (-0.55)) * M_bin
        Mwidth_corr = M_bin_corr[1:] - M_bin_corr[:-1]
        dndM = n_count / volume / Mwidth_corr
        M_bin = M_bin_corr
        """
        M_width = M_bin[1] - M_bin[0]
        dndM = n_count/volume/M_width
        """
        M_c = 0.5 * (M_bin[1:] + M_bin[:-1])
        return cls(mass=M_c, dndM=dndM)

    def get_dndM(self, halo_mass: np.array, **kwargs) -> np.array:
        """
        Get the cumulative halo number density at given
        halo mass thresholds

        Args:
            halo_mass (np.array): mass of the haloes

        Returns:
            np.array: number density of haloes above threshold
        """
        return self.dndM(halo_mass)

    def __call__(self, halo_mass: np.array, **kwargs):
        return self.get_dndM(halo_mass=halo_mass)

    def set_cosmology(self, **kwargs):
        pass


class DarkQuestHaloMassFunction(HaloMassFunction):
    def __init__(
        self, cosmology: Optional[Cosmology] = None, redshift: Optional[float] = None
    ):
        self.hmf = hmf_gp()
        self.current_cosmology = None
        self.current_redshift = None
        if cosmology is not None or redshift is not None:
            self.set_cosmology(cosmology, redshift=redshift)

    def get_dndM(self, halo_mass: np.array, **kwargs) -> np.array:
        return self.dndM(halo_mass)

    def set_cosmology(self, cosmology: Cosmology, redshift: float):
        if cosmology != self.current_cosmology or redshift != self.current_redshift:
            if cosmology != self.current_cosmology:
                self.hmf.set_cosmology(cosmology,)
                self.current_cosmology = cosmology
            hmf_at_m = self.hmf.get_dndM(redshift=redshift)
            self.dndM = ius(self.hmf.Mlist, hmf_at_m, ext=1)
            # this ext is important!
            self.current_redshift = redshift

    def __call__(
        self, halo_mass: np.array, cosmology: Cosmology, redshift: float,
    ):
        self.set_cosmology(
            cosmology, redshift=redshift,
        )
        return self.get_dndM(halo_mass=halo_mass)
