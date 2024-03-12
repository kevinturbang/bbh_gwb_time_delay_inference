import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import gammainc

import numpy as np
from custom_distributions import *
from util import *

from gwBackground import *

logit_std = 2.5
tmp_max = 100.
tmp_min = 2.

##################################################
#######            Likelihood              #######
##################################################

def combined_pop_gwb_cbc_time_delay(sampleDict, injectionDict, rate_file_path, joint_analysis=True, stochasticProds=None):
    """
    Implementation of a Gaussian effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    rate_file_path: str
        Path to the file containing the formation rates, time delays and redshifts
    joint_analysis : bool
        Whether to include the stochastic contribution to the likelihood or not
    stochasticProds: dict
        Dictionary with arrays of frequencies ('freqs'), point estimate spectrum ('Cf') and 
        variance spectrum ('sigma2s') from stochastic search
    """
    
    formationRates, tdelays, zformation, zs = unpack_rate_file(rate_file_path)
    
    if joint_analysis:
        m1s_drawn, m2s_drawn, zs_drawn, p_m1_old, p_m2_old, p_z_old, dEdfs = precompute_omega_weights(stochasticProds['freqs'], N=20000, tmp_min = tmp_min, tmp_max=tmp_max)
    
    # Sample our hyperparameters
    # alpha: Power-law index on primary mass distribution
    # mu_m1: Location of gaussian peak in primary mass distribution
    # sig_m1: Width of gaussian peak
    # f_peak: Fraction of events comprising gaussian peak
    # mMax: Location at which BBH mass distribution tapers off
    # mMin: Lower boundary at which BBH mass distribution tapers off
    # dmMax: Taper width above maximum mass
    # dmMin: Taper width below minimum mass
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # mu: Mean of the chi-effective distribution
    # logsig_chi: Log10 of the chi-effective distribution's standard deviation
    # sig_cost: Standard deviation of the spin angle distribution
    # td_min: minimum time delay
    # metMin_td: metallicity threshold at which the metallicity distribution decreases rapidly
    # lambda_td: slope of the time-delay distribution
    # R20: merger rate amplitude at z=0.2 and m1 = 20 Msun

    logR20 = numpyro.sample("logR20",dist.Uniform(-2,1))
    alpha = numpyro.sample("alpha",dist.Normal(-2,3))
    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,50))
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))
    bq = numpyro.sample("bq",dist.Normal(0,3))
    
    R20 = numpyro.deterministic("R20",10.**logR20)
    
    lambda_td = numpyro.sample("lambda_td",dist.Normal(-1,3))
        
    logit_logmetMin_td = numpyro.sample("logit_logmetMin_td",dist.Normal(0,logit_std))
    logmetMin_td,jac_logmetMin_td = get_value_from_logit(logit_logmetMin_td,-4. ,0.)
    numpyro.deterministic("logmetMin_td",logmetMin_td)
    numpyro.factor("p_logmetMin_td",logit_logmetMin_td**2/(2.*logit_std**2)-jnp.log(jac_logmetMin_td))
    metMin_td = numpyro.deterministic("metMin_td",10.**logmetMin_td)
    
    logit_log_td_min = numpyro.sample("logit_log_td_min",dist.Normal(0,logit_std))
    log_td_min,jac_log_td_min = get_value_from_logit(logit_log_td_min,-3. ,0.)
    numpyro.deterministic("log_td_min",log_td_min)
    numpyro.factor("p_log_td_min",logit_log_td_min**2/(2.*logit_std**2)-jnp.log(jac_log_td_min))
    td_min = numpyro.deterministic("td_min",10.**log_td_min)

    logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
    sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,1.5 ,15.)
    numpyro.deterministic("sig_m1",sig_m1)
    numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))

    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-3. ,0.)
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
    f_peak= numpyro.deterministic("f_peak",10.**log_f_peak)

    logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
    mMax,jac_mMax = get_value_from_logit(logit_mMax,50. ,100.)
    numpyro.deterministic("mMax",mMax)
    numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))

    logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
    log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin, -1. ,0.5)
    numpyro.deterministic("log_dmMin",log_dmMin)
    numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))

    logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))
    log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.5 ,1.5)
    numpyro.deterministic("log_dmMax",log_dmMax)
    numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))

    logit_mu_chi= numpyro.sample("logit_mu_chi",dist.Normal(0,logit_std))
    mu_chi,jac_mu_chi = get_value_from_logit(logit_mu_chi,0. ,1.)
    numpyro.deterministic("mu_chi",mu_chi)
    numpyro.factor("p_mu_chi",logit_mu_chi**2/(2.*logit_std**2)-jnp.log(jac_mu_chi))

    logit_logsig_chi = numpyro.sample("logit_logsig_chi",dist.Normal(0,logit_std))
    logsig_chi,jac_logsig_chi = get_value_from_logit(logit_logsig_chi,-1. ,0.)
    numpyro.deterministic("logsig_chi",logsig_chi)
    numpyro.factor("p_logsig_chi",logit_logsig_chi**2/(2.*logit_std**2)-jnp.log(jac_logsig_chi))

    logit_sig_cost = numpyro.sample("logit_sig_cost",dist.Normal(0,logit_std))
    sig_cost,jac_sig_cost = get_value_from_logit(logit_sig_cost,0.3 ,2.)
    numpyro.deterministic("sig_cost",sig_cost)
    numpyro.factor("p_sig_cost",logit_sig_cost**2/(2.*logit_std**2)-jnp.log(jac_sig_cost))

    # Fixed params
    mu_cost = 1.

    # Merger rate stuff
    fs = gammainc(0.84,(metMin_td**2.)*np.power(10.,0.3*zformation))
    weightedFormationRates = formationRates*fs

    dpdt = jnp.power(tdelays,lambda_td)
    dpdt = jnp.where(tdelays>td_min,dpdt,0)
    dpdt = jnp.where(tdelays<13.5,dpdt,0)
    mergerRate = weightedFormationRates.dot(dpdt)

    # Normalization
    p_m1_norm = massModel(20.,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)
    p_z_norm = jnp.interp(jnp.array([0.2]), zs, mergerRate)[0]

    # Read out found injections
    a1_det = injectionDict['a1']
    a2_det = injectionDict['a2']
    cost1_det = injectionDict['cost1']
    cost2_det = injectionDict['cost2']
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']

    # Compute proposed population weights
    p_m1_det = massModel(m1_det,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq)-tmp_min**(1.+bq))
    p_a1_det = truncatedNormal(a1_det,mu_chi,10.**logsig_chi,0,1)
    p_a2_det = truncatedNormal(a2_det,mu_chi,10.**logsig_chi,0,1)
    p_cost1_det = truncatedNormal(cost1_det,mu_cost,sig_cost,-1,1)
    p_cost2_det = truncatedNormal(cost2_det,mu_cost,sig_cost,-1,1)

    rate_det = jnp.interp(z_det, zs, mergerRate)
    p_z_det = dVdz_det/(1.+z_det)*rate_det/p_z_norm 
    R_pop_det = R20*p_m1_det*p_m2_det*p_z_det*p_a1_det*p_a2_det*p_cost1_det*p_cost2_det

    # Form ratio of proposed weights over draw weights
    inj_weights = R_pop_det/(p_draw/2.)
    
    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute net detection efficiency and add to log-likelihood
    Nexp = jnp.sum(inj_weights)/injectionDict['nTrials']
    numpyro.factor("rate",-Nexp)

    # This function defines the per-event log-likelihood
    # m1_sample: Primary mass posterior samples
    # m2_sample: Secondary mass posterior samples
    # z_sample: Redshift posterior samples
    # dVdz_sample: Differential comoving volume at each sample location
    # a1_sample: Spin magnitude posterior samples
    # a2_sample: Spin magnitude posterior samples
    # cost1_sample: Spin angle posterior samples
    # cost2_sample: Spin angle posterior samples
    # priors: PE priors on each sample

    def logp_cbc(m1_sample,m2_sample,z_sample,dVdz_sample,a1_sample,a2_sample,cost1_sample,cost2_sample,priors):

        # Compute proposed population weights
        p_m1 = massModel(m1_sample,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq)-tmp_min**(1.+bq))
        p_a1 = truncatedNormal(a1_sample,mu_chi,10.**logsig_chi,0,1)
        p_a2 = truncatedNormal(a2_sample,mu_chi,10.**logsig_chi,0,1)
        p_cost1 = truncatedNormal(cost1_sample,mu_cost,sig_cost,-1,1)
        p_cost2 = truncatedNormal(cost2_sample,mu_cost,sig_cost,-1,1)

        rate = jnp.interp(z_sample, zs, mergerRate)
        p_z = dVdz_sample/(1.+z_sample)*rate/p_z_norm
        R_pop = R20*p_m1*p_m2*p_z*p_a1*p_a2*p_cost1*p_cost2

        mc_weights = R_pop/priors
        
        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp_cbc)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z'] for k in sampleDict]), 
                        jnp.array([sampleDict[k]['dVc_dz'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z_prior'] for k in sampleDict]))

    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp_cbc",jnp.sum(log_ps))

    # Stochastic GWB part of the likelihood
    if joint_analysis:
        
        def f_z(z):
            rate = jnp.interp(z, zs, mergerRate)
            rate_final = rate/jnp.sqrt(OmgM*(1.+z)**3.+OmgL)/(1.+z)
            return rate_final
        
        # Read in the stochastic data products
        freqs = stochasticProds['freqs']
        Cf = stochasticProds['Cf']
        sigma2s = stochasticProds['sigma2s']
        
        # Define the log likelihood for the GWB
        def logp_gwb(freqs, Cf, sigma2s):
            p_m2_new = ((1.+bq)*jnp.power(m2s_drawn,bq)/(jnp.power(m1s_drawn,1.+bq)-tmp_min**(1.+bq)))
            p_m1_new = massModel(m1s_drawn,alpha, mu_m1,sig_m1,10.**log_f_peak, mMax, mMin, 10.**log_dmMax,10.**log_dmMin)
            p_z_new = f_z(zs_drawn)
        
            w_i = p_z_new*p_m1_new*p_m2_new/(p_z_old*p_m1_old*p_m2_old)
        
            Omega_spectrum_new = (freqs)*(jnp.einsum("if,i->if",dEdfs,w_i))
            Omega_spectrum_new_avged = 1/rhoC/H0*R20/1e9/year/p_z_norm/p_m1_norm*jnp.mean(Omega_spectrum_new, axis=0)
            Omega_f = Omega_spectrum_new_avged
            diff = Omega_f-Cf
            log_stoch = -0.5*jnp.sum((diff**2/sigma2s))
            return log_stoch
    
        logps_gwb = logp_gwb(freqs, Cf, sigma2s)

        # Add log likelihood for GWB to the total likelihood
        numpyro.factor("logp_gwb", logps_gwb)