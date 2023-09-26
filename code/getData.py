import numpy as np
from astropy.cosmology import Planck15
import astropy.units as u
from custom_distributions import massModel
from scipy.io import loadmat

def getInjections(injectionFile):

    """
    Function to load and preprocess found injections for use in numpyro likelihood functions.

    Returns
    -------
    injectionDict : dict
        Dictionary containing found injections
    """

    injectionDict = np.load(injectionFile,allow_pickle=True)

    for key in injectionDict:
        if key!='nTrials':
            injectionDict[key] = np.array(injectionDict[key])

    return injectionDict

def getSamples(sampleDict_path, sample_limit=1000,bbh_only=True):

    """
    Function to load and preprocess BBH posterior samples for use in numpyro likelihood functions.
    
    Parameters
    ----------
    sample_limit : int or None
        If specified, will randomly downselect posterior samples, returning N=sample_limit samples per event (default None)
    bbh_only : bool
        If true, will exclude samples for BNS, NSBH, and mass-gap events (default True)

    Returns
    -------
    sampleDict : dict
        Dictionary containing posterior samples
    """

    # Dicts with samples:
    sampleDict = np.load(sampleDict_path,allow_pickle=True)

    non_bbh = ['GW170817','S190425z','S190426c','S190814bv','S190917u','S200105ae','S200115j']
    if bbh_only:
        for event in non_bbh:
            print("Removing ",event)
            sampleDict.pop(event)

    for event in sampleDict:
        
        draw_weights = np.ones(sampleDict[event]['m1'].size)/sampleDict[event]['m1'].size

        sampleDict[event]['downselection_Neff'] = np.sum(draw_weights)**2/np.sum(draw_weights**2)

        inds_to_keep = np.random.choice(np.arange(sampleDict[event]['m1'].size),size=sample_limit,replace=True,p=draw_weights)
        for key in sampleDict[event].keys():
            if key!='downselection_Neff':
                sampleDict[event][key] = sampleDict[event][key][inds_to_keep]
        
    return sampleDict

def get_stochastic_dict(file_path, f_high=200.):
    """
    Get stochastic gravitational-wave background data. Assumes a .mat file format, containing
    the point estimate and sigma spectra, as well as the corresponding frequencies.
    
    file_path: str
        Path to the .mat file containing the stochastic results
    f_high: float
        Highest frequency (Hz) to consider for the stochastic contribution.
    """
    matdata = loadmat(file_path)
    Cf = np.array(matdata['ptEst_ff']).reshape(-1)
    sigmas = np.array(matdata['sigma_ff']).reshape(-1)
    freqs = np.array(matdata['freq']).reshape(-1)

    # Select frequencies below f_high
    lowFreqs = freqs<f_high
    freqs = freqs[lowFreqs]
    Cf = Cf[lowFreqs]
    sigma2s = sigmas[lowFreqs]**2.

    goodInds = np.where(Cf==Cf)
    freqs = freqs[goodInds]
    Cf = Cf[goodInds]
    sigma2s = sigma2s[goodInds]

    stochasticDict = {'freqs': freqs, 'Cf': np.real(Cf), 'sigma2s': sigma2s}
    
    return stochasticDict
    

if __name__=="__main__":

    test = getInjections(reweight=True,weighting_function=reweighting_function_arlnm1_q)
    
