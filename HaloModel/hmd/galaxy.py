from dataclasses import dataclass
import numpy as np


# @dataclass
class Galaxy:
    def __init__(
        self,
        logM_min: float = 13.62,
        sigma_logM: float = 0.6915,
        kappa: float = 0.51,
        logM1: float = 14.42,
        alpha: float = 0.9168,
        M_cut: float = 10 ** 12.26,
        M_sat: float = 10 ** 14.87,
        concentration_bias: float = 1.0,
        v_bias_centrals: float = 0.0,
        v_bias_satellites: float = 1.0,
        B_cen: float = 0.,
        B_sat: float = 0.,
        **kwargs,
    ):
        self.M_min = 10 ** logM_min
        self.sigma_logM = sigma_logM
        self.kappa = kappa
        self.M1 = 10 ** logM1
        self.alpha = alpha
        self.M_cut = M_cut
        self.M_sat = M_sat
        self.concentration_bias = concentration_bias
        self.v_bias_centrals = v_bias_centrals
        self.v_bias_satellites = v_bias_satellites
        self.B_cen = B_cen
        self.B_sat = B_sat

    @property
    def logM_min(self,):
        return np.log10(self.M_min)

    @property
    def sigma_logM_sq(self,):
        return self.sigma_logM ** 2

    @property
    def logM1(self,):
        return np.log10(self.M1)


"""
    M_min: float = 10 ** (13.62)
    sigma_logM: float = 0.7
    kappa: float = 0.51
    M1: float = 10 ** 14.43
    alpha: float = 0.92
    M_cut: float = 10 ** 12.26
    M_sat: float = 10 ** 14.87
    concentration_bias: float = 1.0
    v_bias_centrals: float = 0.0
    v_bias_satellites: float = 1.0
"""
