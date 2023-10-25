from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from astropy.cosmology import wCDM
from astropy import units as u
from scipy import integrate
from scipy.misc import derivative
import os 
import sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
from dark_emulator.darkemu.sigmaM import sigmaM_gp
import jax.numpy as jnp

default_emulator_data_path = Path("/cosma6/data/dp004/dc-cues1/emulator/")
default_perturbation_theory_path = Path(
    "/cosma6/data/dp004/dc-cues1/emulator/perturbation_theory/"
)

# TODO: Ask why Onu0 does not match that of Dark Quest


class Cosmology(wCDM):
    def __init__(
        self,
        H0: float,
        Om0: float,
        Ode0: float,
        w0: float,
        wc0: float,
        Ob0: float,
        n_s: float,
        Tcmb0 = 2.7255,
        Neff = 3.046,
        m_nu = 0.06,
        lnAs: float = 3.094,
        name: str = 'default',
    ):
        super().__init__(
            H0=H0,
            Om0=Om0,
            Ode0=Ode0,
            w0=w0,
            Ob0=Ob0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu=m_nu,
            name=name,
        )
        self.wb0 = self.Ob0*self.h**2
        self.wc0 = wc0
        self.lnAs = lnAs
        self.n_s = n_s

        # Temproary disable printing to stdout
        # Prevents printing of "Initialize sigmaM emulator" every f-ing time
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        self.sigma_M_emulator = sigmaM_gp()

        # Restore printing to stdout
        sys.stdout = old_stdout


    @classmethod
    def from_run(
        cls,
        run: int,
        perturbation_theory_path: str = default_perturbation_theory_path,
        emulator_data_path: str = default_emulator_data_path,
        name: Optional[str] = 'dark_quest',
    ):
        cosmo_dict = cls.get_cosmology_for_run(
            run=run, emulator_data_path=default_emulator_data_path
        )
        return cls(
            H0=100.0 * cosmo_dict["h"],
            Om0=cosmo_dict["Om"],
            Ode0=cosmo_dict["Ol"],
            w0=cosmo_dict["w"],
            wc0=cosmo_dict["wc"],
            Ob0=cosmo_dict["wb"] / cosmo_dict["h"] ** 2,
            Tcmb0=2.7255 * u.K,
            Neff=3.046,
            m_nu=[0.0, 0.0, 0.06] * u.eV,
            lnAs=cosmo_dict["lnAs"],
            n_s=cosmo_dict["ns"],
            name=name,
        )
    
    @classmethod
    def from_custom(
        cls,
        run: int,
        perturbation_theory_path: str = default_perturbation_theory_path,
        emulator_data_path: str = default_emulator_data_path,
        name: Optional[str] = 'abacus_v',
    ):
        cosmo_dict = cls.get_cosmology_for_run(
            run=run, emulator_data_path=emulator_data_path
        )
        if np.floor(cosmo_dict["N_eff"]) == 2:
            mnu = [0.0, 0.06] * u.eV
        elif np.floor(cosmo_dict["N_eff"]) == 3:
            mnu = [0.0, 0.0, 0.06] * u.eV
        else:
            raise ValueError("N_eff must be 2 or 3")
        

        return cls(
            H0=100.0 * cosmo_dict["h"],
            Om0=cosmo_dict["Om"],
            Ode0=cosmo_dict["Ol"],
            w0=cosmo_dict["w"],
            wc0=cosmo_dict["wc"],
            Ob0=cosmo_dict["wb"] / cosmo_dict["h"] ** 2,
            Tcmb0=2.7255 * u.K,
            Neff=cosmo_dict["N_eff"],
            m_nu=mnu, #[0.0, 0.0, 0.06] * u.eV,
            lnAs=cosmo_dict["ln1e10As"],
            n_s=cosmo_dict["ns"],
            name=name+str(int(cosmo_dict["version"])),
        )

    @classmethod
    def from_cparams(cls, wb0: float, wc0: float, Ode0: float, lnAs: float, n_s:float, w0: float, **kwargs):
        w_nu0 = 0.00064
        h = jnp.sqrt((wb0+wc0+w_nu0)/(1.-Ode0))
        return cls(
                H0= 100*h,
                Om0=1.-Ode0,
                Ode0=Ode0,
                w0=w0,
                wc0=wc0,
                Ob0=wb0/h**2,
                Tcmb0=2.7255 * u.K,
                Neff=3.046,
                m_nu=[0.0, 0.0, 0.06] * u.eV,
                lnAs=lnAs,
                n_s=n_s,
                name='custom',
        )

    @classmethod
    def get_cosmology_for_run(
        cls, run: int, emulator_data_path: str = default_emulator_data_path
    ) -> dict:
        if run > 100:
            run = 0
        return (
            # pd.read_csv(current_dir + "/cosmological_parameters.dat", sep=" ")
            pd.read_csv(emulator_data_path + "/cosmological_parameters.dat", sep=" ")
            .iloc[run]
            .to_dict()
        )

    @property
    def cparams(self,):
        return [self.wb0, self.wc0, self.Ode0, self.lnAs, self.n_s, self.w0]

    def get_cosmology(self,):
        # for compatibility with dark quest
        wb0 = self.Ob0 * self.h ** 2
        return np.array([wb0, self.wc0, self.Ode0, self.lnAs, self.n_s, self.w0]).reshape(1,6)

    def km_s_to_Mpc_h(self, a: float) -> float:
        z = 1.0 / a - 1.0
        return 1.0 / a / self.efunc(z=z) / 100.0

    def linear_growth_factor(self, redshift) -> float:  # unnormalized
        a = 1.0 / (1 + redshift)
        alpha = -1.0 / (3.0 * self.w0)
        beta = (self.w0 - 1.0) / (2.0 * self.w0)
        gamma = 1.0 - 5.0 / (6.0 * self.w0)
        x = -self.Ode0 / self.Om0 * a ** (-3.0 * self.w0)
        res = integrate.quad(
            lambda t: t ** (beta - 1.0)
            * (1.0 - t) ** (gamma - beta - 1.0)
            * (1.0 - t * x) ** (-alpha),
            0,
            1.0,
        )
        return a * res[0]

    def Dgrowth_from_z(self, redshift) -> float:
        # now normalized to unity at z=0
        return self.linear_growth_factor(redshift=redshift) / self.linear_growth_factor(
            redshift=0
        )


    def Dgrowth_from_a(self, a) -> float:
        # same as above, but now the scale factor "a" as the input
        redshift = 1.0 / a - 1.0
        return self.Dgrowth_from_z(redshift)

    def f_from_a(self, a: float) -> float:
        # computation of the growth rate f, by taking the derivative of D
        return derivative(lambda t: a * np.log(self.Dgrowth_from_a(a=t)), a, dx=1e-6)

    def f_from_z(self, redshift: float) -> float:
        # same as above, but now take the redshift as an argument
        a = 1.0 / (1.0 + redshift)
        return self.f_from_a(a=a)

    def get_sigma8(self, ):
        self.sigma_M_emulator.set_cosmology(cosmo=self)
        rho_crit = self.critical_density(0.) # At redshift 0 
        rho_crit = rho_crit.to(u.solMass/u.Mpc**3*self.h**2).value
        rho_m = self.Om(0.)*rho_crit
        M_8mpc = 4*np.pi/3.*8.**3*rho_m
        return self.sigma_M_emulator.get_sigma(M=M_8mpc)

    def get_approximate_bao(self, redshift):
        h = self.H(redshift).value/100.
        return 147. * h * (self.wc0/0.13)**(-0.25) * (self.wb0/0.024)**(-0.08)



class PlanckCosmology(Cosmology):
    def __init__(self,):
        cosmo_dict = super().get_cosmology_for_run(
            run=101, emulator_data_path=default_emulator_data_path
        )
        return super().__init__(
            H0=100.0 * cosmo_dict["h"],
            Om0=cosmo_dict["Om"],
            Ode0=cosmo_dict["Ol"],
            w0=cosmo_dict["w"],
            wc0=cosmo_dict["wc"],
            Ob0=cosmo_dict["wb"] / cosmo_dict["h"] ** 2,
            Tcmb0=2.7255 * u.K,
            Neff=3.046,
            m_nu=[0.0, 0.0, 0.06] * u.eV,
            lnAs=cosmo_dict["lnAs"],
            n_s=cosmo_dict["ns"],
            name='Planck'
        )

    def get_bias(self, logn: float, snapshot: int,
                perturbation_theory_path: str=default_perturbation_theory_path):
        logn = abs(logn)
        logn_bins = np.arange(2.5, 9.0, 0.5)
        (logn_idx,) = np.where(logn_bins == logn)
        logn_idx = logn_idx[0]
        bias_fid = np.load(perturbation_theory_path / "bias_fid.npy")
        return np.mean(bias_fid[:, snapshot, logn_idx])


