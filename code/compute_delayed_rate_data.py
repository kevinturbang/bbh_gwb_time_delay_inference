import numpy as np
from scipy.interpolate import interp1d
import sys
import os
from constants import *
codeDir = os.path.dirname(os.path.realpath(__file__))

"""
Script to precompute formation and time delay data used in the code.
This file loads in cosmological data from `redshiftData.dat` and computes the following:

`zs`: A 1D array of redshifts matching that in `redshiftData.dat`
`tds`: A 1D array of delay times (units of Gyr) between 0.01 Gyr -- 13.5 Gyr
`formationRedshifts`: A 2D array of formation redshifts corresponding to each combination of *merger* redshifts and time delays.
    The value `formationRedshifts[i,j]` corresponds to the formation redshift of a binary that mergers at zs[i] after 
    a time delay tds[j].
`formationRates`: A 2D array of the SFR (not metallicity weighted) at each of the redshifts in `formationRedshifts`

Data is saved to: `delatedRateData.npy`
"""

# MD rate density parameters
#alpha = 2.7
#beta = 5.6
#zpeak = 1.9

# Vangioni rate density
# doi:10.1093/mnras/stu2600
# End of Section 2.1
z0 = 1.72
a = 2.80
b = 2.46

alpha = 2.7
beta = 2.9
gamma = 5.6

# Define array of redshifts
dz = 0.01
zs = np.arange(0.,10.0,dz)

# Set up a grid of time delays (Gyr)
tds = np.arange(0.005,13.5,0.005)

# Also set up a 2D grid that will hold the **formation redshifts** zf(zm,td)
# corresponding to a merger at z=zm after a time delay td
formationRedshifts = np.zeros((zs.size,tds.size))

# Prepare to save the SFR at these formation redshifts
formationRates = np.zeros((zs.size,tds.size))

# Loop across merger redshifts
for i,zm in enumerate(zs):
    # At each merger redshift, build an interpolator to map from time delays back to a formation redshift
    # First define a reference array of formation redshifts
    # Our resolution is given by solving for the step size dz that will give us time delays < 10 Myr
    dz = (0.003*Gyr)*(1.+0.)*H0*np.sqrt(OmgM+OmgL)
    zf_data = np.arange(zm,15.,dz)               

    # Now compute time delays between the given merger redshift zm and each of the zf's
    timeDelayIntegrands = (1./Gyr)*np.power((1.+zf_data)*H0*np.sqrt(OmgM*np.power(1.+zf_data,3.)+OmgL),-1.)
    td_data = np.insert(np.cumsum(timeDelayIntegrands)*dz,0,0)[:-1]

    # Evaluate formation redshifts corresponding to each time delay in "tds"
    zfs = np.interp(tds,td_data,zf_data)
    formationRedshifts[i,:] = zfs

    # Get the star formation rate at this *formation* redshift
    # Line below is for Madau+ SFR
    formationRates[i,:] = (1+zfs)**alpha/(1+((1+zfs)/beta)**gamma)
    # For Vangioni+ SFR, uncomment lines below
    #formationRates[i,:] = a*np.exp(b*(zfs-z0))/(a-b+b*np.exp(a*(zfs-z0)))
    formationRates[i,zfs!=zfs] = 0
    formationRates[i,zfs>10.] = 0

formationRates[formationRates!=formationRates] = 0.
formationRedshifts[formationRedshifts!=formationRedshifts] = 0.

delayedRateDict = {}
delayedRateDict['formationRates'] = formationRates
delayedRateDict['formationRedshifts'] = formationRedshifts
delayedRateDict['zs'] = zs
delayedRateDict['tds'] = tds

np.save('delayedRateData.npy',delayedRateDict)
