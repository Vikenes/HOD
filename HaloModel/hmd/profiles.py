from scipy import special
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models import NFWProfile, AnalyticDensityProf
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simps
from hmd.cosmology import Cosmology
import hmd.concentration


from dark_emulator.darkemu.sigmaM import sigmaM_gp

# TODO: Generalize to any profile (if doesnt implement analytical fourier transform just compute it numerically)


class FixedCosmologyProfile(AnalyticDensityProf, ABC):
    @abstractmethod
    def dimensionless_mass_density(self, scaled_radius):
        pass

    @abstractmethod
    def inverse_cdf(self, p):
        pass

    def mc_generate_radial_positions(
        self,
        halo_radius: np.array,
        halo_concentration: np.array,
        num_pts=int(1e4),
        seed=None,
        **kwargs
    ):
        # Build lookup table from which to tabulate the inverse cumulative_mass_PDF
        with NumpyRNGContext(seed):
            uniforms = np.random.uniform(0, 1, num_pts)
        # apply inverse CDF
        scaled_radial_positions = self.inverse_cdf(
            uniforms, halo_concentration=halo_concentration, **kwargs
        )
        return scaled_radial_positions * halo_radius

    def mc_generate_radial_velocities(self, velocity_dispersion, seed=None, **kwargs):
        # TODO: improve this with jeans scaling
        # https://halotools.readthedocs.io/en/latest/source_notes/empirical_models/phase_space_models/nfw_profile_source_notes.html#nfw-jeans-velocity-profile-derivations
        # virial_velocities = self.virial_dispersion(total_mass)
        with NumpyRNGContext(seed):
            radial_velocities = np.random.normal(scale=velocity_dispersion)
        return radial_velocities

    @abstractmethod
    def fourier_mass_density(self, k, mass):
        pass


class FixedCosmologyNFW(NFWProfile, FixedCosmologyProfile):
    def __init__(
        self,
        cosmology,
        redshift,
        mdef="200m",
        conc_mass_model="dutton_maccio14",
        sigmaM=None,
    ):
        super().__init__(
            cosmology=cosmology,
            redshift=redshift,
            mdef=mdef,
            conc_mass_model=conc_mass_model,
        )
        self.sigmaM = sigmaM

    def get_f(self, c):
        return np.log(1 + c) - c / (1 + c)

    def fourier_mass_density(self, k, mass, **kwargs):
        """
        https://arxiv.org/pdf/astro-ph/0206508.pdf (Eq 81)

        return shape (c, k)
        """
        conc = self.conc_NFWmodel(
            prim_haloprop=mass,
            cosmology=self.cosmology,
            redshift=self.redshift,
            sigma8=self.cosmology.get_sigma8(),  # self.sigmaM)
            # if self.sigmaM is not None
            # else None,
        )
        radius = self.halo_mass_to_halo_radius(mass) * (1.0 + self.redshift)
        k = k.reshape(1, -1)
        conc = conc.reshape(-1, 1)
        radius = radius.reshape(-1, 1)
        f = self.get_f(c=conc)
        r_s = radius / conc
        si_1pc, ci_1pc = special.sici(k * r_s * (1.0 + conc))
        si, ci = special.sici(k * r_s)
        a_term = np.sin(k * r_s) * (si_1pc - si)
        b_term = np.sin(conc * k * r_s) / (1 + conc) / k / r_s
        c_term = np.cos(k * r_s) * (ci_1pc - ci)
        return 1.0 / f * (a_term - b_term + c_term)

    def inverse_cdf(self, p, halo_concentration, **kwargs):
        """
        https://arxiv.org/pdf/1805.09550.pdf (Eq 6)
        """
        p *= self.get_f(c=halo_concentration)
        return (
            -(1.0 / np.real(special.lambertw(-np.exp(-p - 1)))) - 1
        ) / halo_concentration

    def virial_dispersion(self, mass):
        # sqrt(1+ redshift) to transform radius to comoving coordinates
        # from physical
        return self.virial_velocity(mass) / np.sqrt(2) / np.sqrt(1 + self.redshift)

    def get_norm_radial_profile(
        self,
        r: np.array,
        radial_profile: np.array,
        radius: np.array,
        n_integrand: int = 2000,
        log_r_min: float = -5.0,
    ):
        """
        Compute norm for profile truncated at radius
        """
        r_norm = np.logspace(log_r_min, np.log10(max(radius)), n_integrand)
        radial_profile = np.array([ius(r, xi, ext=3)(r_norm) for xi in radial_profile])
        dlog_r = np.log(r_norm[1]) - np.log(r_norm[0])
        integrand_norm = (
            4.0 * np.pi * r_norm.reshape(1, -1) ** 3 * (1.0 + radial_profile)
        )
        integrand_norm = np.where(
            r_norm.reshape(1, -1) > radius.reshape(-1, 1), 0.0, integrand_norm
        )
        return simps(integrand_norm, dx=dlog_r, axis=-1)

    def real_mass_density(
        self, r, mass,
    ):
        conc = self.conc_NFWmodel(
            prim_haloprop=mass,
            cosmology=self.cosmology,
            redshift=self.redshift,
            sigma8=self.cosmology.get_sigma8(self.sigmaM)
            if self.sigmaM is not None
            else None,
        )
        radius = self.halo_mass_to_halo_radius(mass) * (1.0 + self.redshift)
        xi_hm = np.array(
            [self.mass_density(r, mass=m, conc=conc[i],) for i, m in enumerate(mass)]
        )
        norm = self.get_norm_radial_profile(r=r, radial_profile=xi_hm, radius=radius,)
        xi_hm = np.where(r.reshape(1, -1) > radius.reshape(-1, 1), 0.0, xi_hm,)
        return xi_hm / norm.reshape(-1, 1)

    def mc_generate_radial_positions(
        self, halo_concentration, num_pts=int(1e4), seed=None, **kwargs
    ):
        """
        conc = self.conc_NFWmodel(
            prim_haloprop=mass,
            cosmology=self.cosmology,
            redshift=self.redshift,
            sigma8=self.cosmology.get_sigma8(self.sigmaM)
            if self.sigmaM is not None
            else None,
        )
        """
        return super().mc_generate_radial_positions(
            num_pts=num_pts, halo_radius=1.0, halo_concentration=halo_concentration,
        )


class NFW:
    def __init__(
        self, conc_mass_model=hmd.concentration.diemer15, sigmaM=sigmaM_gp(), mdef="200m",
    ):
        self.mdef = mdef
        self.conc_mass_model = conc_mass_model
        self.sigmaM = sigmaM

    def get_profile(self, cosmology: Cosmology, redshift: float):
        return FixedCosmologyNFW(
            cosmology=cosmology,
            redshift=redshift,
            mdef=self.mdef,
            conc_mass_model=self.conc_mass_model,
            sigmaM=self.sigmaM,
        )



