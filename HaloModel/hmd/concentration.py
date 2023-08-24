import numpy as np
import jax.numpy as jnp
from hmd.utils import simpson
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import UnivariateSpline as us


DELTA_COLLAPSE = 1.68647

"""The linear overdensity threshold for halo collapse according to the spherical top-hat collapse 

model (`Gunn & Gott 1972 <http://adsabs.harvard.edu/abs/1972ApJ...176....1G>`_). This number 

corresponds to :math:`3/5 (3\pi/2)^{2/3}` and is modified very slightly in a non-EdS universe."""


def ludlow(
    halo_mass, redshift, cosmology,
):
    delta_sc_z = 1.686
    xi = 1.0 / (halo_mass / 1e10)
    sig_z_fit = 22.26 * xi ** (0.292) / (1.0 + 1.53 * xi ** 0.275 + 3.36 * xi ** 0.198)
    nu_z = delta_sc_z / sig_z_fit
    z_plus = 1.0 + redshift
    nu0 = (
        4.135
        - 0.564 * z_plus
        - 0.210 * z_plus ** 2
        + 0.0557 * z_plus ** 3
        - 0.00348 * z_plus ** 4
    )
    c0 = 3.395 * z_plus ** (-0.215)
    beta = 0.307 * z_plus ** (0.540)
    gamma1 = 0.628 * z_plus ** (-0.047)
    gamma2 = 0.317 * z_plus ** (-0.893)
    return (
        c0
        * (nu_z / nu0) ** (-gamma1)
        * (1.0 + (nu_z / nu0) ** (1.0 / beta)) ** (-beta * (gamma2 - gamma1))
    )


def mu(x):
    return jnp.log1p(x) - x / (1.0 + x)


def get_NFW_fundamental_parameters(M, c, z, mdef, cosmology):
    rs = 1000.0 * cosmology.halo_mass_to_halo_radius(mass=M, z=z, mdef=mdef) / c
    rhos = M / rs ** 3 / 4.0 / jnp.pi / mu(c)
    return rhos, rs


def get_NFW_xDelta(rhos, density_threshold):
    table_x = jnp.logspace(4.0, -4.0, 1000)
    table_y = mu(table_x) * 3.0 / table_x ** 3
    x_interpolator = ius(table_y, table_x, k=3,)
    knots = x_interpolator.get_knots()
    x_interpolator_min = knots[0]
    x_interpolator_max = knots[-1]
    y = density_threshold / rhos
    if jnp.min(y) < x_interpolator_min:
        raise Exception(
            "Requested overdensity %.2e cannot be evaluated for scale density %.2e, out of range."
            % (jnp.min(y), x_interpolator_min)
        )
    if jnp.max(y) > x_interpolator_max:
        raise Exception(
            "Requested overdensity %.2e cannot be evaluated for scale density %.2e, out of range."
            % (jnp.max(y), x_interpolator_max)
        )
    return x_interpolator(y)


def evolveSO(
    M_i, c_i, z_i, mdef_i, z_f, mdef_f, cosmology,
):
    Rnew = jnp.zeros_like(M_i)
    cnew = jnp.zeros_like(M_i)
    rhos, rs = get_NFW_fundamental_parameters(
        M_i, c_i, z_i, mdef_i, cosmology=cosmology
    )
    density_threshold = cosmology.density_threshold(z=z_f, mdef=mdef_f) / 1.0e9
    cnew = get_NFW_xDelta(rhos, density_threshold)
    Rnew = rs * cnew
    Mnew = cosmology.halo_radius_to_halo_mass(radius=Rnew / 1000.0, z=z_f, mdef=mdef_f)
    return Mnew, Rnew, cnew


def change_mass_definition(M, c, z, mdef_in, mdef_out, cosmology):
    return evolveSO(M, c, z, mdef_in, z, mdef_out, cosmology=cosmology)


def lagrangianR(cosmology, halo_mass):
    return (3.0 * halo_mass / 4.0 / jnp.pi / cosmology.rho_m(0.0) / 1e9) ** (1.0 / 3.0)


def wtop_hat(x):
    r"""Return the tophat function math:`W(x) = 3 (\sin(x)-x\cos(x))/x^{3}` to :math:`\mathcal{O}(x^{10})`."""
    return 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x ** 3


def sigma_r(r, pk, cosmology, redshift, kmin=1e-6, kmax=1e2):
    def integrand(logk):
        k = jnp.exp(logk).reshape(1, -1)
        return (
            pk(k=k.reshape(-1), cosmology=cosmology, redshift=redshift,).reshape(
                1, -1
            )
            * (wtop_hat(r.reshape(-1, 1) * k) * k) ** 2
            * k
        )

    logk_range = jnp.linspace(jnp.log(kmin), jnp.log(kmax), 100)
    dk = logk_range[1] - logk_range[0]
    sigmasq = 1.0 / 2.0 / jnp.pi ** 2 * simpson(integrand(logk_range), dx=dk, axis=-1)
    return jnp.sqrt(sigmasq)


def get_peak_height(
    cosmology, pk, M, z,
):
    R = lagrangianR(cosmology, M)
    sigma = sigma_r(r=R, pk=pk, cosmology=cosmology, redshift=z,)
    return DELTA_COLLAPSE / sigma


def _diemer15_k_R(cosmology, halo_mass):
    # green
    R = lagrangianR(cosmology, halo_mass)
    return 2.0 * jnp.pi / R


def _diemer15_n(pk_linear, cosmology, redshift, k_R):
    k_min = jnp.min(k_R) * 0.9
    k_max = jnp.max(k_R) * 1.1
    logk = jnp.arange(jnp.log10(k_min), jnp.log10(k_max), 0.001)
    Pk = pk_linear(k=10 ** logk, cosmology=cosmology,redshift=redshift)
    # NEED SMOOTHING OVER k, otherwise noisy derivative
    interp = us(logk, jnp.log10(Pk))
    return interp(jnp.log10(k_R), nu=1)


def _diemer15_n_fromM(pk_linear, cosmology, redshift, halo_mass):
    k_R = _diemer15_k_R(cosmology, halo_mass)
    return _diemer15_n(pk_linear, cosmology, redshift, k_R)


def _diemer15(nu, n):
    DIEMER15_MEDIAN_PHI_0 = 6.58
    DIEMER15_MEDIAN_PHI_1 = 1.27
    DIEMER15_MEDIAN_ETA_0 = 7.28
    DIEMER15_MEDIAN_ETA_1 = 1.56
    DIEMER15_MEDIAN_ALPHA = 1.08
    DIEMER15_MEDIAN_BETA = 1.77
    floor = DIEMER15_MEDIAN_PHI_0 + n * DIEMER15_MEDIAN_PHI_1
    nu0 = DIEMER15_MEDIAN_ETA_0 + n * DIEMER15_MEDIAN_ETA_1
    alpha = DIEMER15_MEDIAN_ALPHA
    beta = DIEMER15_MEDIAN_BETA
    return 0.5 * floor * ((nu0 / nu) ** alpha + (nu / nu0) ** beta)


def jax_diemer15(
    halo_mass, redshift, cosmology, pk, mdef="200m",
):
    n = _diemer15_n_fromM(
        pk_linear=pk, cosmology=cosmology, redshift=redshift, halo_mass=halo_mass
    )
    nu = get_peak_height(cosmology=cosmology, pk=pk, M=halo_mass, z=redshift)
    c_200c = _diemer15(nu, n)
    if mdef == "200c":
        return c_200c
    M_200m, R_200m, c_200m = change_mass_definition(
        halo_mass, c_200c, redshift, "200c", "200m", cosmology=cosmology,
    )
    return ius(M_200m, c_200m)(halo_mass)


def diemer15(prim_haloprop, redshift, cosmology, sigma8, allow_tabulation=True):
    from colossus.cosmology import cosmology as colcosmology
    from colossus.halo import concentration

    n_s = cosmology.n_s if hasattr(cosmology, "n_s") else 0.96
    sigma8 = sigma8 if sigma8 is not None else 0.8
    _ = colcosmology.fromAstropy(cosmology, sigma8, n_s, name="custom")
    if len(prim_haloprop) > 100.0 and allow_tabulation:
        interpolate_mass = np.logspace(11.0, np.log10(np.max(prim_haloprop)), 300)
        concentration_table = concentration.concentration(
            interpolate_mass, mdef="200m", z=redshift, model="diemer15"
        )
        return us(np.log10(interpolate_mass), concentration_table, ext=3)(
            np.log10(prim_haloprop)
        )
    else:
        return concentration.concentration(
            prim_haloprop, mdef="200m", z=redshift, model="diemer15"
        )
