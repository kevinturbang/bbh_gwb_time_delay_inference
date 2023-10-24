from scipy.io import loadmat
import numpy as np
from gwBackground import *

def precompute_omega_weights(freqs, N=20000, tmp_min=2, tmp_max=100):
    """
    Function to precompute the 

    Parameters
    ----------

    freqs: array-like
        Array of frequencies for which to compute the energy spectrum radiated by the binaries.

    N: int
        Number of samples to generate, and to eventually perform the Monte-Carlo average on. Several
        values of this number have been tested, showing good convergence for the N=20000 case.

    tmp_min: float
        Minimum value used to draw the primary mass uniformly.

    tmp_max: float
        Maximum value used to draw the primary mass uniformly.
    Returns
    -------
    m1s_drawn: array-like
        Array containing the uniformly drawn values of the primary mass
    m2s_drawn: array-like
        Array containing the uniformly drawn values of the secindary mass
    zs_drawn: array-like
        Array containing the uniformly drawn values of the redshifts
    p_m1_old: array-like
        Array containing the probability to draw the drawn m1 values, assuming a uniform distribution
    p_m2_old: array-like
        Array containing the probability to draw the drawn m2 values, assuming a uniform distribution
    p_z_old: array-like
        Array containing the probability to draw the drawn redshift values, assuming a uniform distribution
    dEdfs: array-like
        Array containing the energy density spectrum emitted by the binary with parameters given by the drawn parameters above.
    """

    m1s_drawn = np.random.uniform(tmp_min, tmp_max, size=N)

    c_m2s = np.random.uniform(size=int(N))
    m2s_drawn = tmp_min**(1.)+c_m2s*(m1s_drawn**(1.)-tmp_min**(1.))
    
    zs_drawn = np.random.uniform(0,10,size=N)

    dEdfs = np.array([dEdf(m1s_drawn[ii]+m2s_drawn[ii],freqs*(1+zs_drawn[ii]),eta=m2s_drawn[ii]/m1s_drawn[ii]/(1+m2s_drawn[ii]/m1s_drawn[ii])**2) for ii in range(N)])

    p_m1_old = 1/(tmp_max-tmp_min)*np.ones(N)
    p_z_old = 1/(10-0)*np.ones(N)
    p_m2_old = 1/(m1s_drawn-tmp_min)
    
    dEdfs = np.array([dEdf(m1s_drawn[ii]+m2s_drawn[ii],freqs*(1+zs_drawn[ii]),eta=m2s_drawn[ii]/m1s_drawn[ii]/(1+m2s_drawn[ii]/m1s_drawn[ii])**2) for ii in range(N)])

    return  m1s_drawn, m2s_drawn, zs_drawn, p_m1_old, p_m2_old, p_z_old, dEdfs


def unpack_rate_file(filename):
    """
    Unpacks the file containing the formation rates, time-delays, and formation redshifts.

    Parameters
    ----------

    filename: str
        Full path to the file containing the information about the rate.

    Returns
    -------

    formationRates: array-like
        Array containing the formation rates
    tdelays: array-like
        Array containing the time-delays
    zformation: array-like
        Array containing the foramtion redshifts
    """
    rateData = np.load(filename, allow_pickle=True)[()]
    formationRates = rateData['formationRates']
    tdelays = rateData['tds']
    zformation = rateData['zs']
    return formationRates, tdelays, zformation

def get_value_from_logit(logit_x,x_min,x_max):
    """
    Helper function to sample uniformly between values `x_min` and `x_max`. This allows
    to circumvent issues related to samples close to the edges when using a NUTS sampler,
    which relies on gradients.

    Parameters
    ----------

    logit_x: float
        Value of x in logit space.
    x_min: float
        Lower bound of uniform prior.
    x_max: float
        Upper bound of uniform prior.

    Returns
    -------

    x: float
        Value of the uniformly drawn parameter in regular space.

    dlogit_dx: float
        Jacobian derivative used to take into account the coordinate transform to go from logit space
        to regular space. This will be used in the likelihood to account for the above.


    """
    exp_logit = jnp.exp(logit_x)
    x = (exp_logit*x_max + x_min)/(1.+exp_logit)
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)

    return x,dlogit_dx