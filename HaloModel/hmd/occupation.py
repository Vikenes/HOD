import numpy as np
from scipy import special
from abc import ABC, abstractmethod
from hmd.galaxy import Galaxy


# Inherit from halotools


class Occupation(ABC):
    @abstractmethod
    def get_n(self, halo_property, galaxy: Galaxy):
        pass


class Zheng07Centrals(Occupation):
    def get_n(self, halo_mass: np.array, galaxy: Galaxy, **kwargs) -> np.array:
        """Compute the number of central galaxies as a function of host halo mass.

        Args:
            halo_mass (np.array): halo_mass
            galaxy: object containing galaxy parameters

        Equation (G2) at arXiv:1811.09504

        Returns:
            np.array:
        """
        return 0.5 * special.erfc(
            np.log10(galaxy.M_min / halo_mass) / galaxy.sigma_logM
        )

    def get_velocity_dispersion(
        self, halo_dispersions: np.array, galaxy: Galaxy
    ) -> np.array:
        return galaxy.v_bias_centrals * halo_dispersions


class Zheng07Sats(Occupation):
    def lambda_sat(self, halo_mass: np.array, galaxy: Galaxy, **kwargs) -> np.array:
        mask = halo_mass > galaxy.kappa * galaxy.M_min
        lambda_value = np.zeros_like(halo_mass)
        lambda_value[mask] = ((halo_mass[mask] - galaxy.kappa * galaxy.M_min) / galaxy.M1) ** galaxy.alpha
        return lambda_value
        '''
        return np.where(
            halo_mass > galaxy.kappa * galaxy.M_min,
            ((halo_mass - galaxy.kappa * galaxy.M_min) / galaxy.M1) ** galaxy.alpha,
            0.0,
        )
        '''

    def get_n(
        self, halo_mass: np.array, n_centrals: np.array, galaxy: Galaxy, **kwargs
    ) -> np.array:
        """Compute the number of satellite galaxies as a function of host
        halo mass.

        Equation (G4) at arXiv:1811.09504

        Args:
            halo_mass (np.array): halo_mass
            n_centrals (np.array): number of central galaxies in that
            host halo mass
            galaxy: object contaiing galaxy parameters

        Returns:
            np.array:
        """
        return np.where(
            halo_mass > galaxy.kappa * galaxy.M_min,
            n_centrals * self.lambda_sat(halo_mass=halo_mass, galaxy=galaxy),
            0.0,
        )

    def get_velocity_dispersion(
        self, halo_dispersions: np.array, galaxy: Galaxy
    ) -> np.array:
        return galaxy.v_bias_satellites * halo_dispersions

    def get_concentration(
        self, halo_concentrations: np.array, galaxy: Galaxy
    ) -> np.array:
        return galaxy.concentration_bias * halo_concentrations


class ABCentrals(Occupation):
    def get_n(
        self, halo_mass: np.array, halo_rank: np.array, galaxy: Galaxy, **kwargs
    ) -> np.array:
        """
        https://arxiv.org/pdf/2007.05545.pdf Eq(7)
        """
        M_min = 10 ** (np.log10(galaxy.M_min) + galaxy.B_cen * (halo_rank - 0.5))
        return 0.5 * special.erfc(np.log10(M_min / halo_mass) / galaxy.sigma_logM)

    def get_velocity_dispersion(
        self, halo_dispersions: np.array, galaxy: Galaxy
    ) -> np.array:
        return galaxy.v_bias_centrals * halo_dispersions


class ABSatellites(Occupation):
    def lambda_sat(
        self, halo_mass: np.array, halo_rank: np.array, galaxy: Galaxy
    ) -> np.array:
        M_cut = galaxy.kappa * galaxy.M_min
        M_cut = 10 ** (np.log10(M_cut) + galaxy.B_sat * (halo_rank - 0.5))
        M1 = 10 ** (np.log10(galaxy.M1) + galaxy.B_sat * (halo_rank - 0.5))
        return np.where(
            halo_mass > M_cut, ((halo_mass - M_cut) / M1) ** galaxy.alpha, 0.0,
        )

    def get_n(
        self,
        halo_mass: np.array,
        halo_rank: np.array,
        n_centrals: np.array,
        galaxy: Galaxy,
    ) -> np.array:
        """Compute the number of satellite galaxies as a function of host
        halo mass.

        https://arxiv.org/pdf/2007.05545.pdf Eq(8)

        Args:
            halo_mass (np.array): halo_mass
            n_centrals (np.array): number of central galaxies in that
            host halo mass
            galaxy: object contaiing galaxy parameters

        Returns:
            np.array:
        """
        return n_centrals * self.lambda_sat(
            halo_mass=halo_mass, halo_rank=halo_rank, galaxy=galaxy
        )

    def get_velocity_dispersion(
        self, halo_dispersions: np.array, galaxy: Galaxy
    ) -> np.array:
        return galaxy.v_bias_satellites * halo_dispersions

    def get_concentration(
        self, halo_concentrations: np.array, galaxy: Galaxy
    ) -> np.array:
        return galaxy.concentration_bias * halo_concentrations


class AemulusCentrals(Occupation):
    def get_n(
        self, halo_mass: np.array, halo_delta: np.array, galaxy: Galaxy, **kwargs
    ) -> np.array:
        """
        """
        M_min = galaxy.M_min * (
            1.0
            + galaxy.f_env * special.erf((delta - galaxy.delta_env) / galaxy.sigma_env)
        )
        return 0.5 * special.erfc(np.log10(M_min / halo_mass) / galaxy.sigma_logM)

    def get_velocity_dispersion(
        self, halo_dispersions: np.array, galaxy: Galaxy
    ) -> np.array:
        return galaxy.v_bias_centrals * halo_dispersions


class AemulusSatellites(Occupation):
    def lambda_sat(self, halo_mass: np.array, galaxy: Galaxy) -> np.array:
        return (halo_mass / galaxy.M_sat) ** galaxy.alpha * np.exp(
            -galaxy.M_cut / halo_mass
        )

    def get_n(
        self, halo_mass: np.array, n_centrals: np.array, galaxy: Galaxy
    ) -> np.array:
        """Compute the number of satellite galaxies as a function of host
        halo mass.

        Equation (7) at arXiv:1804.05867

        Args:
            halo_mass (np.array): halo_mass
            n_centrals (np.array): number of central galaxies in that
            host halo mass
            galaxy: object contaiing galaxy parameters

        Returns:
            np.array:
        """
        return n_centrals * self.lambda_sat(halo_mass=halo_mass, galaxy=galaxy)

    def get_velocity_dispersion(
        self, halo_dispersions: np.array, galaxy: Galaxy
    ) -> np.array:
        return galaxy.v_bias_satellites * halo_dispersions

    def get_concentration(
        self, halo_concentrations: np.array, galaxy: Galaxy
    ) -> np.array:
        return galaxy.concentration_bias * halo_concentrations
