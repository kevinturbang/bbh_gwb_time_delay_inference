import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from constants import *
import jax.numpy as jnp
from jax.scipy.special import erf
import jax
jax.config.update("jax_enable_x64", True)

tmp_max = 100.
tmp_min = 2.

codeDir = os.path.dirname(os.path.realpath(__file__))

def v(Mtot,f):
    return np.array([(np.pi*Mtot*MsunToSec*f)**(1./3.), (np.pi*Mtot*MsunToSec*f)**(2./3.), (np.pi*Mtot*MsunToSec*f)])

def dEdf(Mtot,freqs,eta=0.25,inspiralOnly=False,PN=True):

    """
    Function to compute the energy spectrum radiated by a CBC
    
    INPUTS
    Mtot: Total mass in units of Msun
    freqs: Array of frequencies at which we want to evaluate dEdf
    eta: Reduced mass ratio. Defaults to 0.25 (equal mass)
    inspiralOnly: If True, will return only energy radiated through inspiral

    """

    chi = 0.

    # Initialize energy density
    dEdf_spectrum = np.zeros(freqs.shape)

    if inspiralOnly:

        # If inspiral only (used for BNS), cut off at the ISCO
        fMerge = 2.*c**3./(6.*np.sqrt(6.)*2.*np.pi*G*Mtot*Msun)
        inspiral = freqs<fMerge
        dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)

    else:

        if PN:

            # Waveform model from Ajith+ 2011 (10.1103/PhysRevLett.106.241101)
            
            # PN corrections to break frequencies bounding different waveform regimes
            # See Eq. 2 and Table 1
            eta_arr = np.array([eta,eta*eta,eta*eta*eta])
            chi_arr = np.array([1,chi,chi*chi]).T
            fM_corrections = np.array([[0.6437,0.827,-0.2706],[-0.05822,-3.935,0.],[-7.092,0.,0.]])
            fR_corrections = np.array([[0.1469,-0.1228,-0.02609],[-0.0249,0.1701,0.],[2.325,0.,0.]])
            fC_corrections = np.array([[-0.1331,-0.08172,0.1451],[-0.2714,0.1279,0.],[4.922,0.,0.]])
            sig_corrections = np.array([[-0.4098,-0.03523,0.1008],[1.829,-0.02017,0.],[-2.87,0.,0.]])

            # Define frequencies
            # See Eq. 2 and Table 1
            fMerge = (1. - 4.455*(1.-chi)**0.217 + 3.521*(1.-chi)**0.26 + eta_arr.dot(fM_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
            fRing = (0.5 - 0.315*(1.-chi)**0.3 + eta_arr.dot(fR_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
            fCut = (0.3236 + 0.04894*chi + 0.01346*chi*chi + eta_arr.dot(fC_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)
            sigma = (0.25*(1.-chi)**0.45 - 0.1575*(1.-chi)**0.75 + eta_arr.dot(sig_corrections).dot(chi_arr))/(np.pi*Mtot*MsunToSec)

            # Identify piecewise components
            inspiral = freqs<fMerge
            merger = (freqs>=fMerge)*(freqs<fRing)
            ringdown = (freqs>=fRing)*(freqs<fCut)

            # Define PN amplitude corrections
            # See Eq. 1 and following text
            alpha = np.array([0., -323./224. + 451.*eta/168., (27./8.-11.*eta/6.)*chi])
            eps = np.array([1.4547*chi-1.8897, -1.8153*chi+1.6557, 0.])
            vs = v(Mtot,freqs)

            # Compute multiplicative scale factors to enforce continuity of dEdf across boundaries
            # Note that w_m and w_r are the ratios (inspiral/merger) and (merger/ringdown), as defined below
            v_m = v(Mtot,fMerge)
            v_r = v(Mtot,fRing)
            w_m = np.power(fMerge,-1./3.)*np.power(1.+alpha.dot(v_m),2.)/(np.power(fMerge,2./3.)*np.power(1.+eps.dot(v_m),2.)/fMerge)
            w_r = (w_m*np.power(fRing,2./3.)*np.power(1.+eps.dot(v_r),2.)/fMerge)/(np.square(fRing)/(fMerge*fRing**(4./3.)))

            # Energy spectrum
            dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)*np.power(1.+alpha.dot(vs[:,inspiral]),2.)
            dEdf_spectrum[merger] = w_m*np.power(freqs[merger],2./3.)*np.power(1.+eps.dot(vs[:,merger]),2.)/fMerge
            dEdf_spectrum[ringdown] = w_r*np.square(freqs[ringdown]/(1.+np.square((freqs[ringdown]-fRing)/(sigma/2.))))/(fMerge*fRing**(4./3.))

        else:

            # Waveform model from Ajith+ 2008 (10.1103/PhysRevD.77.104017)
            # Define IMR parameters
            # See Eq. 4.19 and Table 1
            fMerge = (0.29740*eta**2. + 0.044810*eta + 0.095560)/(np.pi*Mtot*MsunToSec)
            fRing = (0.59411*eta**2. + 0.089794*eta + 0.19111)/(np.pi*Mtot*MsunToSec)
            fCut = (0.84845*eta**2. + 0.12828*eta + 0.27299)/(np.pi*Mtot*MsunToSec)
            sigma = (0.50801*eta**2. + 0.077515*eta + 0.022369)/(np.pi*Mtot*MsunToSec)

            # Identify piecewise components
            inspiral = freqs<fMerge
            merger = (freqs>=fMerge)*(freqs<fRing)
            ringdown = (freqs>=fRing)*(freqs<fCut)

            # Energy spectrum
            dEdf_spectrum[inspiral] = np.power(freqs[inspiral],-1./3.)
            dEdf_spectrum[merger] = np.power(freqs[merger],2./3.)/fMerge
            dEdf_spectrum[ringdown] = np.square(freqs[ringdown]/(1.+np.square((freqs[ringdown]-fRing)/(sigma/2.))))/(fMerge*fRing**(4./3.))

    # Normalization
    Mc = np.power(eta,3./5.)*Mtot*Msun
    amp = np.power(G*np.pi,2./3.)*np.power(Mc,5./3.)/3.

    return amp*dEdf_spectrum

class OmegaGW(object):

    """
    Base class used to compute the stochastic energy density of a CBC population.
    To make the evaluation as fast as possible, we'll implement integration over redshift and mass distribution
    via array multiplication. Different mass distributions are imposed by specifying probability weights in mass space.
    """

    def __init__(self,ref_mMin,ref_mMax,ref_zs,fmax,inspiralOnly,Mtots=[],qs=[],gridSize=(70,65)):

        """
        Initializes class by setting up a grid of masses and mass ratios, and evaluating the radiated energy spectrum dE/df
        across this grid. This allows us to precompute dE/df, only evaluating it *once*. When computing a *population-averaged*
        energy spectrum, we will take a weighted sum across this grid. Weights are imposed by redefining `self.probs`, implemented
        in child classes via the `setProbs()` function.

        INPUTS
        ref_mMin: Minimum component mass to consider in mass grid
        ref_mMax: Maximum component mass to consider in mass grid
        ref_zs: Redshift array across which we will integrate to compute Omega(f)
        fMax: Maximum detector-frame frequency to consider
        inspiralOnly: If true, restrict dE/df to the energy radiated before the ISCO
        """

        # Save reference grid of redshifts
        self.ref_zs = ref_zs

        # Initialize grid of masses over which we'll compute dE/df
        # In particular, we'll grid log-uniformly in total mass and uniformly in mass ratio q
        self.ref_mMin = ref_mMin
        self.ref_mMax = ref_mMax

        if len(Mtots)!=0:
            self.ref_Mtots = Mtots
        else:
            self.ref_Mtots = np.logspace(np.log10(2.*self.ref_mMin),np.log10(2.*self.ref_mMax),gridSize[0])

        if len(qs)!=0:
            self.ref_qs = qs
        else:
            qMin = max(0.05,ref_mMin/ref_mMax)
            self.ref_qs = np.linspace(qMin,1,gridSize[1])

        # Grid
        self.Mtots_2d,self.qs_2d = np.meshgrid(self.ref_Mtots,self.ref_qs)

        # Compute the component masses and reduced mass ratios at each grid point
        self.m1s_2d = self.Mtots_2d/(1.+self.qs_2d)
        self.m2s_2d = self.qs_2d*self.Mtots_2d/(1.+self.qs_2d)
        self.ref_etas = self.ref_qs/(1.+self.ref_qs)/(1.+self.ref_qs)

        # Array of frequencies at which dE/df will be evaluated for each point in our mass grid
        self.ref_freqs = np.logspace(np.log10(10),np.log10(fmax),250)

        # Remember that we need to consider binaries placed across a *range* of redshifts.
        # For a given system mass, we need to know not just dE/df(f), but the *redshifted* spectrum dE/df(f*(1+z))
        # at both difference frequencies f and merger redshifts z

        # Set up a 2D array of source-frame frequencies f(1+z)
        # i.e. self.ref_redshiftedFreqs[i,j] corresponds to self.ref_freqs[j]*(1.+self.ref_zs[i])
        self.ref_redshiftedFreqs = np.array([self.ref_freqs*(1.+z) for z in self.ref_zs])

        # Now evaluate the energy spectrum at each of these source-frame frequencies, for every point in our mass grid
        # self.ref_energySpectra[i,j,k,:] is the energy contribution from a CBC with reduced mass ratio self.ref_etas[i],
        # total mass self.ref_Mtots[j], and at redshift self.ref_zs[k].
        self.ref_energySpectra = np.array([[dEdf(M,self.ref_redshiftedFreqs,eta=eta,inspiralOnly=inspiralOnly) for M in self.ref_Mtots] for eta in self.ref_etas])

        # Initialize weights for mass grid
        self.probs = np.ones(self.Mtots_2d.shape)
        self.probs /= np.sum(self.probs)

    def eval(self,dRdV, dRdV02, targetFreqs):

        """
        Given a prescription for the local merger rate and its evolution over redshift, compute Omega(f)

        INPUTS
        dRdV: Arbitrarily normalized merger rate density as a function of redshift. Should correspond to self.ref_zs
        dRdV02: Merger rate density evaluated at z=0.2 and m1=20 solar masses
        targetFreqs: Array of frequencies at which we want Omega(f). Must be above 10 and below fMax
        """

        dRdV_norm = dRdV/dRdV02 #(R20 dependence is already in probs)

        # Compute weighted average of energy-density spectrum, integrated over total mass and mass ratio space
        # The result is a 2D array, with dedf[i,j] the population-averaged energy contributed at detector-frame
        # frequency i by binaries at redshift j

        dedf = jnp.transpose(jnp.tensordot(self.probs,jnp.asarray(self.ref_energySpectra),axes=2))
        
        # Redshift integrand
        R_invE = dRdV_norm/jnp.sqrt(OmgM*(1.+self.ref_zs)**3.+OmgL)/(1.+self.ref_zs)
        # Integrate over redshifts via a dot product between dedf and the redshift-dependent R_invE
        dz = self.ref_zs[1]-self.ref_zs[0]
        Omg_spectrum = (self.ref_freqs/rhoC/H0)*dedf.dot(R_invE)*dz 
        # Interpolate onto desired frequencies and return
        final_Omg_spectrum = jnp.interp(targetFreqs,self.ref_freqs,Omg_spectrum,left=0.,right=0.)
        return final_Omg_spectrum

def massModel(m1,alpha,mu_m1,sig_m1,f_peak,mMax,mMin,dmMax,dmMin):

    """
    Baseline primary mass model, described as a mixture between a power law
    and gaussian, with exponential tapering functions at high and low masses

    Parameters
    ----------
    m1 : array or float
        Primary masses at which to evaluate probability densities
    alpha : float
        Power-law index
    mu_m1 : float
        Location of possible Gaussian peak
    sig_m1 : float
        Stanard deviation of possible Gaussian peak
    f_peak : float
        Approximate fraction of events contained within Gaussian peak (not exact due to tapering)
    mMax : float
        Location at which high-mass tapering begins
    mMin : float
        Location at which low-mass tapering begins
    dmMax : float
        Scale width of high-mass tapering function
    dmMin : float
        Scale width of low-mass tapering function

    Returns
    -------
    p_m1s : jax.numpy.array
        Unnormalized array of probability densities
    """

    # Define power-law and peak
    p_m1_pl = (1.+alpha)*m1**(alpha)/(tmp_max**(1.+alpha) - tmp_min**(1.+alpha))
    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    # Apply filters to combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter

class OmegaGW_BBH(OmegaGW):

    """
    Subclass of `OmegaGW`, used to compute energy density due to BBHs under a "Broken Power Law" mass model

    Implements a function `setProbs` to fix the weights used in integrating over the mass grid to compute
    a population averaged energy spectrum
    """

    def __init__(self,ref_mMin,ref_mMax,ref_zs,Mtots=[],qs=[],gridSize=(70,65)):
        super(OmegaGW_BBH,self).__init__(ref_mMin,ref_mMax,ref_zs,3000,False,Mtots=Mtots,qs=qs,gridSize=gridSize)

    def setProbs_plPeak(self,mMin,mMax,lmbda,mu_peak,sig_peak,frac_peak,bq,R20, dmMax,dmMin):
        # Jacobian with which to convert integration over d(lnM)dq to d(m1)d(m2)
        probs_jacobian = self.Mtots_2d**2./(1.+self.qs_2d)**2.

        # p(m1) is power-law + peak
        probs_m1 =  massModel(self.m1s_2d,lmbda, mu_peak,sig_peak,frac_peak,mMax,mMin,dmMax,dmMin)
 
        probs_m1 = jnp.where(self.m1s_2d>=tmp_max,0,probs_m1)

        #normalized to m1=20
        probs_norm_m1_20 = massModel(20,lmbda, mu_peak,sig_peak,frac_peak,mMax,mMin,dmMax,dmMin)

        # Probability on secondary mass
        probs_m2 = (1.+bq)*jnp.power(self.m2s_2d,bq)/(jnp.power(self.m1s_2d,1.+bq)-tmp_min**(1.+bq))

        probs_m2 = jnp.where(self.m2s_2d<=tmp_min,0,probs_m2)
        
        # Convert to number per Mpc^3 per sec
        R20 = R20/1e9/year

        # Combine and set
        probs = R20*probs_jacobian*probs_m1/probs_norm_m1_20*probs_m2
        probs *= (np.log(self.ref_Mtots[1])-np.log(self.ref_Mtots[0]))*(self.ref_qs[1]-self.ref_qs[0])
        self.probs = probs

if __name__=="__main__":

    freqs = np.arange(10.,400.,1)
    testObject = OmegaGW_BBH(2.,20.,np.linspace(0,8,100))

    """
    R0 = 1.
    mMin = 5.
    lmbda1 = -2.3
    lmbda2 = -6.
    m0 = 35.
    bq = 1.5
    dRdV = np.ones(100)

    e1 = testObject.evalOriginal(R0,dRdV,freqs)

    testObject.setProbs()
    e2 = testObject.eval(R0,dRdV,freqs)
    """






