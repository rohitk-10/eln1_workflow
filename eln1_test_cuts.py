# Commands to select directory based on hostname
from socket import gethostname

if gethostname() == 'colonsay':
    path_start = '/disk1/rohitk/ELN1_project/'
elif gethostname() == 'rohitk-elitebook':
    path_start = '/home/rohitk/Documents/PhD/Year1/ELN1_project/'

#################################################
# Add the path of useful functions at the start
import sys
sys.path.append(path_start+'basic_functions')
from useful_functions import return_hist_par, varstat, latest_dir, logspace_bins
# import overlapping_area as ova
from overlapping_area import (isinpan, isinukidss, isinSWIRE, isinSERVS)
##################################################

import numpy as np
from matplotlib import pyplot as plt
import glob

# For catalog matching
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import match_coordinates_sky
from astropy.coordinates import search_around_sky

plt.style.use('publish')

from astropy.table import Table

##################################################

"""
A simple version of the flowchart to see what source numbers we get
"""

# Definition of the cuts to make
c_large = 15.            # Objects larger than this value are "large" (15'')
c_bright = 10*1e-3            # Objects brighter than this value are "bright" (10mJy)
c_nn = 45.                    # Objects with NN separation greater than this are "isolated" (45'')
c_4nn = 45.                   # Objects with 4th NN within distance are "clustered" (45'')

# Read in the ML output on all overlapping area sources
mlfinal = Table.read(path_start + "OCT17_ELAIS_im/maxl_test/full_runs/26_10_2018_2/ML_RUN_fin_overlap.fits")

# LR threshold (currently hard-coded)
lr_th = 0.5

overlap_indx = []
rect_indx = []

for i in range(len(mlfinal)):
    if ((isinpan(mlfinal['RA'][i],mlfinal['DEC'][i])) and
        (isinukidss(mlfinal['RA'][i],mlfinal['DEC'][i])) and
        (isinSWIRE(mlfinal['RA'][i],mlfinal['DEC'][i]))):

        overlap_indx.append(mlfinal['Source_id'][i])

# Filter to get soures within overlapping area and above threshold
final_ov = mlfinal[(overlap_indx)]
# final_ov = final_ov[(final_ov["lr_fin"] < lr_th)]
low_th = final_ov["lr_fin"] < lr_th

maj = final_ov["Maj"] * 3600.
maj_n, maj_e, maj_c = return_hist_par(0.1, maj)

fig = plt.figure()
plt.hist(maj, maj_e, histtype='step', lw=1.5)
plt.xlabel(r'$Maj $' + ' [arcsec]')

# Number of large sources
large_ind = maj > c_large
print("Sources with maj > {0} arcsec: {1} : {2}% ".format(c_large, np.sum(large_ind), np.sum(large_ind)/len(overlap_indx)*100))

# Number of small sources
nlarge_ind = maj <= c_large

nl_co = SkyCoord(final_ov["RA"], final_ov["DEC"], unit='deg', frame='icrs')

# Match with itself to find nearest neighbour separation
indx, sep2d, _ = match_coordinates_sky(nl_co, nl_co, nthneighbor=2)

# Find the small and isolated sources
sm_iso = ((maj <= c_large) & (sep2d > c_nn*u.arcsec))
print("Small Sources with NN sep > {0} arcsec: {1} : {2}% ".format(c_nn, np.sum(sm_iso), np.sum(sm_iso)/len(overlap_indx)*100))


# Plot of LR vs. source size
fig = plt.figure()
plt.scatter(maj[low_th], np.log10(final_ov["lr_fin"][low_th]), s=4, color='red',
            facecolor='None', edgecolor='red')
# plt.plot(maj, np.log10(final_ov["lr_fin"]), '.', markersize=1.) 
plt.xlabel('Major axis size [arcsec]')
plt.ylabel(r'$log(LR)$')
plt.xscale('log')


# Plot of LR distribution
lr = final_ov["lr_fin"][low_th]

fig = plt.figure()
# _, lr_e, lr_c = return_hist_par(1, lr)
lr_e = logspace_bins(np.nanmin(lr), np.nanmax(lr), 0.1)

plt.hist(np.log10(lr), lr_e, histtype='step', lw=1.)
plt.xlabel('$log(LR)$')
plt.ylabel('$N$')
# plt.savefig('test_cuts/log_lr_dist.pdf')


# Make a plot of the LR vs. the NN distance?
fig = plt.figure()
plt.scatter(sep2d.arcsec[low_th], np.log10(lr), s=7, color='orange', alpha=0.4)
plt.xlabel("NN Separation [arcsec]")
plt.ylabel(r'$log(LR)$')

plt.show()
