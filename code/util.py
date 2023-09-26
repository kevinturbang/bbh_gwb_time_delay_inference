from scipy.io import loadmat
import numpy as np
from gwBackground import *

def precompute_omega_weights(freqs, N=20000, tmp_min=2, tmp_max=100):

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
    rateData = np.load(filename, allow_pickle=True)[()]
    formationRates = rateData['formationRates']
    tdelays = rateData['tds']
    zformation = rateData['zs']
    return formationRates, tdelays, zformation

def get_value_from_logit(logit_x,x_min,x_max):

    exp_logit = jnp.exp(logit_x)
    x = (exp_logit*x_max + x_min)/(1.+exp_logit)
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)

    return x,dlogit_dx