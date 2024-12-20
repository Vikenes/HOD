import jax.numpy as jnp
from jax import vmap, jit
from functools import partial, wraps


@partial(jit, static_argnums=(0, 1, 2, 3))
def slice_array(idx_start, idx_end, step, axis, x):
    return x.take(indices=jnp.arange(idx_start, idx_end, step), axis=axis)


@partial(jit, static_argnums=(2, 3,))
def simps(y, dx, n, axis=-1):
    if n % 2 != 1:
        raise ValueError("len(y) must be an odd integer.")
    return (
        dx
        / 3
        * jnp.sum(
            slice_array(0, n - 1, 2, axis, y)
            + 4 * slice_array(1, n, 2, axis, y)
            + slice_array(2, n, 2, axis, y),
            axis=0,
        )
    )

def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def _basic_simpson(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
    slice2 = tupleset(slice_all, axis, slice(start + 2, stop + 2, step))

    if x is None:  # Even-spaced Simpson's rule.
        result = jnp.sum(y[slice0] + 4 * y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = jnp.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
        h0 = h[sl0]
        h1 = h[sl1]
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = h0 / h1
        tmp = (
            hsum
            / 6.0
            * (
                y[slice0] * (2 - 1.0 / h0divh1)
                + y[slice1] * (hsum * hsum / hprod)
                + y[slice2] * (2 - h0divh1)
            )
        )
        result = jnp.sum(tmp, axis=axis)
    return result


def simpson(y, x=None, dx=1.0, axis=-1, even="avg"):
    """
    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule. If x is None, spacing of dx is assumed.
    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals. The parameter 'even' controls how this is handled.
    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : float, optional
        Spacing of integration points along axis of `x`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.
    even : str {'avg', 'first', 'last'}, optional
        'avg' : Average two results:1) use the first N-2 intervals with
                  a trapezoidal rule on the last interval and 2) use the last
                  N-2 intervals with a trapezoidal rule on the first interval.
        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.
        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.
    See Also
    --------
    quad: adaptive quadrature using QUADPACK
    romberg: adaptive Romberg quadrature
    quadrature: adaptive Gaussian quadrature
    fixed_quad: fixed-order Gaussian quadrature
    dblquad: double integrals
    tplquad: triple integrals
    romb: integrators for sampled data
    cumulative_trapezoid: cumulative integration for sampled data
    ode: ODE integrators
    odeint: ODE integrators
    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less. If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.
    Examples
    --------
    >>> from scipy import integrate
    >>> x = np.arange(0, 10)
    >>> y = np.arange(0, 10)
    >>> integrate.simpson(y, x)
    40.5
    >>> y = np.power(x, 3)
    >>> integrate.simpson(y, x)
    1642.5
    >>> integrate.quad(lambda x: x**3, 0, 9)[0]
    1640.25
    >>> integrate.simpson(y, x, even='first')
    1644.5
    """
    y = jnp.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = jnp.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the " "same as y.")
        if x.shape[axis] != N:
            raise ValueError(
                "If given, length of x along axis must be the " "same as y."
            )
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice1 = (slice(None),) * nd
        slice2 = (slice(None),) * nd
        if even not in ["avg", "last", "first"]:
            raise ValueError("Parameter 'even' must be " "'avg', 'last', or 'first'.")
        # Compute using Simpson's rule on first intervals
        if even in ["avg", "first"]:
            slice1 = tupleset(slice1, axis, -1)
            slice2 = tupleset(slice2, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
            result = _basic_simpson(y, 0, N - 3, x, dx, axis)
        # Compute using Simpson's rule on last set of intervals
        if even in ["avg", "last"]:
            slice1 = tupleset(slice1, axis, 0)
            slice2 = tupleset(slice2, axis, 1)
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val += 0.5 * first_dx * (y[slice2] + y[slice1])
            result += _basic_simpson(y, 1, N - 2, x, dx, axis)
        if even == "avg":
            val /= 2.0
            result /= 2.0
        result = result + val
    else:
        result = _basic_simpson(y, 0, N - 2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result


