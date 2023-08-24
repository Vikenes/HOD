import numpy as np
from scipy.integrate import simps
from jax_fht.cosmology import FFTLog
from hmd import HaloMassFunction, Occupation

# from mimicus.utils import simpson


class HaloModel:
    def __init__(
        self,
        logM_min: float = 12.0,
        logM_max: float = 15.9,
        n_M_bins_1h: int = 50,
        n_M_bins_2h: int = 16,
        fft_num: int = 1,
        fft_logrmin_1h: float = -5.0,
        fft_logrmax_1h: float = 3.0,
        fft_logrmin_2h: float = -3.0,
        fft_logrmax_2h: float = 3.0,
    ):
        self.M_max = 10 ** logM_max
        self.M_min = 10 ** logM_min
        self.halo_mass_1h = np.logspace(logM_min, logM_max, n_M_bins_1h)
        self.log10halo_mass_1h = np.log10(self.halo_mass_1h)

        self.dlog_halo_mass_1h = np.log(self.halo_mass_1h[1]) - np.log(
            self.halo_mass_1h[0]
        )
        self.halo_mass_2h = np.logspace(logM_min, logM_max, n_M_bins_2h)
        self.dlog_halo_mass_2h = np.log(self.halo_mass_2h[1]) - np.log(
            self.halo_mass_2h[0]
        )

        self.do_integration = simps
        self.fft_num = fft_num
        self.fft_logrmin_1h = fft_logrmin_1h
        self.fft_logrmax_1h = fft_logrmax_1h
        self.fft_logrmin_2h = fft_logrmin_2h
        self.fft_logrmax_2h = fft_logrmax_2h
        self.fft_1h = FFTLog(
            fft_num, log_r_min=fft_logrmin_1h, log_r_max=fft_logrmax_1h, kr=1.0
        )
        self.fft_2h = FFTLog(
            fft_num, log_r_min=fft_logrmin_2h, log_r_max=fft_logrmax_2h, kr=1.0
        )

    def get_galaxy_number_density(self, dndM, n_centrals, n_satellites) -> float:
        """Compute the number density of galaxies for a set of HOD parameters

        Equation (G5) at arXiv:1811.09504

        Args:

        Returns:
            float: number density of galaxies
        """
        return self.do_integration(
            dndM * (n_centrals + n_satellites) * self.halo_mass_1h,
            dx=self.dlog_halo_mass_1h,
        )

    def get_centrals_number_density(
        self, dndM, n_centrals,
    ):
        return self.do_integration(
            dndM * n_centrals * self.halo_mass_1h, dx=self.dlog_halo_mass_1h,
        )

    def get_satellites_number_density(self, dndM, n_satellites):
        return self.do_integration(
            dndM * n_satellites * self.halo_mass_1h, dx=self.dlog_halo_mass_1h,
        )
