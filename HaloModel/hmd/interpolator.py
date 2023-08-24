import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from itertools import combinations_with_replacement
from scipy.integrate import simps
from hmd import HaloMassFunction
from hmd.extrapolator import Extrapolator

# TODO: deal with extrapolation (linear theory re-weighting)


class Interpolator:
    def __init__(
        self,
        r: np.array,
        logn: np.array,
        observable: np.array,
        hmf: HaloMassFunction,
        extrapolator: Extrapolator = None,
    ):
        """
        Interpolate over number density measurements, useful to convert
        density thresholds into mass bins

        Args:
            logn (np.array): log number densities of the measurements
            observable (np.array): observable, measured on those logn
            hmf (HaloMassFunction): halo mass function object
        """
        self.r = r
        self.logn = logn
        self.n = 10 ** self.logn
        self.observable = observable
        self.hmf = hmf
        self.extrapolator = extrapolator


class Interpolator1D(Interpolator):
    def __init__(
        self,
        r: np.array,
        logn: np.array,
        observable: np.array,
        hmf: HaloMassFunction,
        extrapolator: Extrapolator = None,
    ):
        super().__init__(
            r=r, logn=logn, observable=observable, hmf=hmf, extrapolator=extrapolator
        )

    def get_for_mass(self, mass: float, r: np.array = None, **kwargs) -> np.array:
        """
        Get observable on mass bin

        Args:
            mass (float): mass

        Returns:
            np.array: observable
        """
        n_mass = self.hmf.convert_mass_to_density(mass)
        observable_shape = self.observable.shape[1:]
        len_n = len(self.n)
        observable = self.observable.reshape((len_n, -1))
        differential = self.n.reshape(-1, 1) * observable
        auto_mass = np.array(
            [
                ius(self.n, differential[:, i]).derivative()(n_mass)
                for i in range(observable.shape[-1])
            ]
        )
        auto_mass = auto_mass.reshape(observable_shape)
        if r is not None:
            if len(auto_mass.shape) == 2:
                return np.array([ius(self.r, am)(r) for am in auto_mass])
            elif len(auto_mass.shape) > 2:
                raise ValueError
            return ius(self.r, auto_mass)(r)
        return auto_mass


class Interpolator2D(Interpolator):
    def __init__(
        self,
        r: np.array,
        logn: np.array,
        observable: np.array,
        hmf: HaloMassFunction,
        logn_extrapolate_min: float = -8.0,
        extrapolator: Extrapolator = None,
    ):
        super().__init__(
            r=r, logn=logn, observable=observable, hmf=hmf, extrapolator=extrapolator
        )
        if extrapolator is not None:
            n = np.array(list(combinations_with_replacement(self.n, r=2)))
            n_non_linear = n[
                (n[:, 0] >= extrapolator.min_n) & (n[:, 1] >= extrapolator.min_n)
            ]
            self.observable = extrapolator(
                n=n, n_non_linear=n_non_linear, y_non_linear=self.observable,
            )
        self.n = self.n[::-1]
        self.observable = self.convert_to_symmetric_matrix(
            len(self.n), self.observable
        )[::-1, ::-1]
        self.differential = (
            self.n.reshape(-1, 1, 1) * self.n.reshape(1, -1, 1) * self.observable
        )
        self.interpolators = [
            rbs(self.n, self.n, self.differential[:, :, i],)
            for i in range(self.observable.shape[-1])
        ]

    def get_auto_mass_by_derivative(
        self, n_mass_left: float, n_mass_right: float, r: np.array = None
    ) -> np.array:

        auto_mass = np.array(
            [
                interpolator.ev(n_mass_left, n_mass_right, dx=1, dy=1)
                for interpolator in self.interpolators
            ]
        )
        if r is not None:
            return ius(self.r, auto_mass)(r)
        return auto_mass

    def get_for_mass(
        self, mass_left: float, mass_right: float, r: np.array = None
    ) -> np.array:
        """
        Get observable on mass bins

        Args:
            mass_left (float): mass
            mass_right (float): mass
            r: radial separation

        Returns:
            np.array: observable
        """
        n_mass_left = abs(self.hmf.convert_mass_to_density(mass_left))
        n_mass_right = abs(self.hmf.convert_mass_to_density(mass_right))
        return self.get_auto_mass_by_derivative(
            n_mass_left=n_mass_left, n_mass_right=n_mass_right, r=r,
        )

    def convert_to_symmetric_matrix(
        self, n_halo_mass: int, obs_hh: np.array
    ) -> np.array:
        """
        Given lower diagonal of obs_hh, convert it to a symmetric matrix
        (to avoid computing all combinations of (m1,m2))

        Args:
            n_halo_mass (int): number of halo mass bins
            obs_hh (np.array): lower diagonal

        Returns:
            np.array:
                symmetric observable
        """
        n_r = obs_hh.shape[1]
        indices = np.triu_indices((n_halo_mass))
        triu = np.zeros((n_halo_mass, n_halo_mass, n_r,))
        triu[indices] = obs_hh
        return np.where(triu, triu, triu.swapaxes(1, 0))

    def get_xi_pk_for_mass_array(self, halo_mass, r, fft, **kwargs):
        xi_hh = np.zeros((len(halo_mass), len(halo_mass), len(r)))
        pk_hh = np.zeros((len(halo_mass), len(halo_mass), len(fft.k)))
        for (i, left_mass) in enumerate(halo_mass):
            for (j, right_mass) in enumerate(halo_mass):
                if j >= i:
                    xi_hh[i, j] = self.get_for_mass(
                        mass_left=left_mass, mass_right=right_mass, r=r,
                    )
                    pk_hh[i, j] = fft.xi2pk(ius(r, xi_hh[i, j], ext=3)(fft.r))
                else:
                    xi_hh[i, j] = xi_hh[j, i]
                    pk_hh[i, j] = pk_hh[j, i]
        return xi_hh, pk_hh
