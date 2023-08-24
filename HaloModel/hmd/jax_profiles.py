from scipy import special
import jax.numpy as jnp
from hmd.concentration import jax_diemer15

class JaxNFW:
    def __init__(
        self, 
        conc_mass_model=jax_diemer15, 
        mdef='200m',
    ):
        self.mdef= mdef
        self.conc_mass_model = conc_mass_model
        
        from mimicus.linear import PkLinear
        self.pk_linear =  PkLinear()

    def get_f(self, c):
        return jnp.log(1 + c) - c / (1 + c)

    def fourier_mass_density(self, k, mass, cosmology, redshift):
        """
        https://arxiv.org/pdf/astro-ph/0206508.pdf (Eq 81)

        return shape (c, k)
        """
        conc = self.conc_mass_model(
            halo_mass=mass,
            cosmology=cosmology,
            redshift=redshift,
            pk=self.pk_linear,
        )
        radius = cosmology.halo_mass_to_halo_radius(mass, mdef=self.mdef, z=redshift) * (1.0 + redshift)
        k = k.reshape(1, -1)
        conc = conc.reshape(-1, 1)
        radius = radius.reshape(-1, 1)
        f = self.get_f(c=conc)
        r_s = radius / conc
        si_1pc, ci_1pc = special.sici(k * r_s * (1.0 + conc))
        si, ci = special.sici(k * r_s)
        a_term = jnp.sin(k * r_s) * (si_1pc - si)
        b_term = jnp.sin(conc * k * r_s) / (1 + conc) / k / r_s
        c_term = jnp.cos(k * r_s) * (ci_1pc - ci)
        return 1.0 / f * (a_term - b_term + c_term)


