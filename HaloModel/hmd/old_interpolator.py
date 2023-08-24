import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.integrate import simps
from hmd import HaloMassFunction
#from hmd.extrapolator import LinearExtrapolator

# TODO: deal with extrapolation (linear theory re-weighting)



class Interpolator:
    def __init__(
        self,
        r: np.array,
        logn: np.array,
        observable: np.array,
        hmf: HaloMassFunction,
        extrapolator = None,
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
        extrapolator = None,
    ):
        super().__init__(
            r=r, logn=logn, observable=observable, hmf=hmf, extrapolator=extrapolator
        )

    def get_xi_at_mass(self, n_splits: int, halo_mass: np.array, fft):
        number_densities = np.array(
            [self.hmf.convert_mass_to_density(mass) for mass in halo_mass]
        )
        if self.extrapolator is not None:
            mask = number_densities > self.extrapolator.min_n
        else:
            mask = number_densities >= np.min(number_densities)
        non_linear_halo_mass = halo_mass[mask]
        linear_halo_mass = halo_mass[~mask]
        xi_h = np.zeros((n_splits, len(non_linear_halo_mass), len(self.r)))
        pk_h = np.zeros((n_splits, len(non_linear_halo_mass), len(fft.k)))
        for i in range(n_splits):
            for (j, mass) in enumerate(non_linear_halo_mass):
                xi_h[i, j] = self.get_auto_mass(mass=mass, r=self.r,)
                pk_h[i, j] = fft.xi2pk(ius(self.r, xi_h[i, j], ext=3)(fft.r))
        if self.extrapolator is not None:
            xi_h = self.extrapolator.extrapolate_auto_mass(
                non_linear_mass=non_linear_halo_mass,
                y_non_linear=xi_h,
                linear_mass=linear_halo_mass,
            )
            pk_h = self.extrapolator.extrapolate_auto_mass(
                non_linear_mass=non_linear_halo_mass,
                y_non_linear=pk_h,
                linear_mass=linear_halo_mass,
            )
        return xi_h, pk_h

    def get_auto_mass(self, mass: float, r: np.array = None) -> np.array:
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
        order=2,
        smoothing=None,
        extrapolator = None,
        pair_weighting: bool = False,
        weight =None,
    ):
        super().__init__(
            r=r, logn=logn, observable=observable, hmf=hmf, extrapolator=extrapolator
        )
        differential = (
            self.n.reshape(-1, 1, 1) * self.n.reshape(1, -1, 1) * self.observable
        )
        if pair_weighting:
            differential *= weight #1.0 + xi_hh_n1n2
        self.interpolators = [
            rbs(self.n, self.n, differential[:, :, i], kx=order, ky=order, s=smoothing)
            for i in range(self.observable.shape[-1])
        ]

    def get_xi_at_mass(self, halo_mass, fft=None, return_pk: bool = False):
        number_densities = np.array(
            [self.hmf.convert_mass_to_density(mass) for mass in halo_mass]
        )
        if self.extrapolator is not None:
            mask = number_densities > self.extrapolator.min_n
        else:
            mask = number_densities >= np.min(number_densities)
        non_linear_halo_mass = halo_mass[mask]
        linear_halo_mass = halo_mass[~mask]
        xi_hh = np.zeros(
            (len(non_linear_halo_mass), len(non_linear_halo_mass), len(self.r))
        )
        pk_hh = np.zeros(
            (len(non_linear_halo_mass), len(non_linear_halo_mass), len(fft.k))
        )
        for (i, left_mass) in enumerate(non_linear_halo_mass):
            for (j, right_mass) in enumerate(non_linear_halo_mass):
                if j >= i:
                    xi_hh[i, j] = self.get_auto_mass(
                        mass_left=left_mass, mass_right=right_mass
                    )
                    pk_hh[i, j] = fft.xi2pk(ius(self.r, xi_hh[i, j], ext=3)(fft.r))
                else:
                    xi_hh[i, j] = xi_hh[j, i]
                    pk_hh[i, j] = pk_hh[j, i]
        if self.extrapolator is not None:
            xi_hh = self.extrapolator.extrapolate_auto_mass(
                non_linear_mass=non_linear_halo_mass,
                y_non_linear=xi_hh,
                linear_mass=linear_halo_mass,
            )
            pk_hh = self.extrapolator.extrapolate_auto_mass(
                non_linear_mass=non_linear_halo_mass,
                y_non_linear=pk_hh,
                linear_mass=linear_halo_mass,
            )
        return xi_hh, pk_hh

    def get_auto_mass(
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

    def get_auto_mass_range(
        self, logM_min: float, logM_max: float, r: np.array = None,
    ):
        n_M_bins = 20
        halo_mass = np.logspace(logM_min, logM_max, n_M_bins)
        dlog_halo_mass = np.log(halo_mass[1]) - np.log(halo_mass[0])
        n_h = simps(self.hmf.dndM(halo_mass)*halo_mass, dx=dlog_halo_mass)

        obs = np.zeros((len(halo_mass), len(halo_mass), len(self.r)))
        for (i, left_mass) in enumerate(halo_mass):
            for (j, right_mass) in enumerate(halo_mass):
                if j >= i:
                    obs[i, j] = self.get_auto_mass(
                        mass_left=left_mass, mass_right=right_mass
                    )
                else:
                    obs[i, j] = obs[j, i]
        obs /= n_h**2
        obs_int = simps(
            simps(
                (halo_mass * self.hmf.dndM(halo_mass)).reshape(-1, 1, 1)
                * (halo_mass * self.hmf.dndM(halo_mass)).reshape(1, -1, 1)
                * obs,
                axis=0,
                dx=dlog_halo_mass,
            ),
            axis=0,
            dx=dlog_halo_mass,
        )
        if r is not None:
            return ius(
                    self.r[np.logical_not(np.isnan(obs_int))],
                    obs_int[np.logical_not(np.isnan(obs_int))],
            )(r)
        return obs_int
