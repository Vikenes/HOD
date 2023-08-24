# from collections import namedtuple
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Cosmology:
    def __init__(
        self,
        wb: float = 0.02225,
        wc: float = 0.1198,
        Ode: float = 0.6844,
        lnAs: float = 3.094,
        n_s: float = 0.9645,
        w: float = -1.0,
    ):
        self.wb = wb
        self.wc = wc
        self.Ode = Ode
        self.lnAs = lnAs
        self.n_s = n_s
        self.w = w

    def tree_flatten(self):
        params = (
            self.wb,
            self.wc,
            self.Ode,
            self.lnAs,
            self.n_s,
            self.w,
        )
        return (params, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Retrieve base parameters
        wb, wc, Ode, lnAs, n_s, w = children
        return cls(wb=wb, wc=wc, Ode=Ode, lnAs=lnAs, n_s=n_s, w=w,)


"""
Cosmology = namedtuple(
        'Cosmology',
        ['wb','wc','Ode','lnAs','n_s','w'],
        defaults=(0.02225, 0.1198,0.6844,3.094,0.9645,-1.),
)
"""
