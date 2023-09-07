import numpy as np
from scipy.special import erf
import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import Planck15

def dVdz(zs):

    c = 299792458 # m/s
    H_0 = 67900.0 # m/s/MPc
    Omega_M = 0.3065 # unitless
    Omega_Lambda = 1.0-Omega_M
    year = 365.25*24.*3600

    cosmo = Planck15.clone(name='cosmo',Om0=Omega_M,H0=H_0/1e3,Tcmb0=0.)
    return 4.*np.pi*cosmo.differential_comoving_volume(zs).to(u.Gpc**3/u.sr).value

def redshiftGrid(zMax,nPoints):

    zs = np.linspace(0,zMax,nPoints)
    dVc_dz = dVdz(zs)

    return zs,dVc_dz

if __name__=="__main__":
    print(redshiftGrid(2.3,300))
