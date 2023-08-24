from dq import Cosmology
from pytest import approx

def test__inputs_outputs():
    c1 = Cosmology.from_run(run=50)

    c2 = Cosmology.from_cparams(*c1.cparams)
    assert c1.cparams == c2.cparams
    assert c1.h == approx(c2.h,rel=1.e-4)
    assert c1.H0.value == approx(c2.H0.value,rel=1.e-4)
    assert c1.Om0  == approx(c2.Om0,rel=1.e-4)
    assert c1.sigma8 == c2.sigma8
