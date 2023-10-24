import numpyro
import numpy as np
nChains = 1  #1 is enough for testing
numpyro.set_host_device_count(nChains)
#numpyro.set_platform('gpu') ##Uncomment if able to run on GPUs
from numpyro.infer import NUTS,MCMC
from jax import random
import arviz as az
from likelihoods import combined_pop_gwb_cbc_time_delay
from getData import *
from get_cosmo import *

stochasticDict = get_stochastic_dict("../input/Omega_measurement_undetectable_O5_MD.mat.mat", f_high=200)

# Get dictionaries holding injections and posterior samples
injectionDict = getInjections("../input/injectionDict_FAR_1_in_1.pickle")
sampleDict = getSamples("../input/sampleDict_FAR_1_in_1_yr.pickle", sample_limit=2000)

# Set up NUTS sampler over our likelihood
kernel = NUTS(combined_pop_gwb_cbc_time_delay)
#Try to increase warmup stage steps to avoid divergences later on
mcmc = MCMC(kernel,num_warmup=500,num_samples=3000,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(118)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict, rate_file_path="../input/delayedRateDataMD.npy", joint_analysis=True, stochasticProds=stochasticDict)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"../data/RUNS/posteriors_Aplus_undetectable_GWB.cdf")

