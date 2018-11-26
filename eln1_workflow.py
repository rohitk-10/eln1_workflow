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

# Read in the ML output on all overlapping area sources
# mlfinal = Table.read(path_start + "OCT17_ELAIS_im/maxl_test/full_runs/26_10_2018_2/ML_RUN_fin_overlap.fits")
mlfinal = Table.read("/home/rohitk/Documents/PhD/Year1/ELN1_project/OCT17_ELAIS_im/maxl_test/full_runs/26_11_2018_1/ML_RUN_fin_overlap_srl.fits")

# LR threshold (currently hard-coded)
lr_th = 0.5


def get_overlap_sources(lofar_cat):
    """
    Return Source_id columns of LOFAR sources within overlapping area of multiwavelength coverage
    """
    overlapping_sources = []
    overlapping_bool = []
    for i in range(len(lofar_cat)):
        if ((isinpan(lofar_cat['RA'][i],lofar_cat['DEC'][i])) and (isinukidss(lofar_cat['RA'][i],lofar_cat['DEC'][i])) and
            (isinSWIRE(lofar_cat['RA'][i], lofar_cat['DEC'][i]))):
            overlapping_sources.append(i)
            overlapping_bool.append(True)
        else:
            overlapping_bool.append(False)

    return overlapping_sources, overlapping_bool


def percent(number_of_sources):
    """
    Return percentage of sources w.r.t. base_sources
    """
    return number_of_sources * 100. / base_sources


# Get Source_id's of overlapping sources
overlap_indx, overlap_bool = get_overlap_sources(mlfinal)
# Set this as the baseline of "All" sources
base_sources = len(overlap_indx)

# Filter to get soures within overlapping area and above threshold
mlfinal_ov = mlfinal[(overlap_indx)]

low_th = mlfinal_ov["lr_fin"] < lr_th

# Definition of the cuts to make
cuts = dict()
cuts["large"] = 15.         # Large size cut
cuts["bright"] = 10*1e-3    # Bright cut
cuts["nnsep"] = 45.         # NN separation (arcsec)
cuts["nth_nn"] = 4          # n number of NNs allowed for source to be "non-clustered"
cuts["nth_nnsep"] = 45.     # nth_nn NNs within this separation allowed for source to be "non-clustered"

# Empty list to keep track of overall numbers
lgz_tot = []
lrid_tot = []
prefilt_tot = []

# Dictionary to keep track of the sources in different blocks
decision_block = dict()

# Get all the large sources
mlfinal_ov["large"] = np.nan
large = mlfinal_ov["Maj"]*3600. > cuts["large"]

decision_block["large"] = np.sum(large)
decision_block["nlarge"] = np.sum(~large)
mlfinal_ov["large"][large] = 1.

# Add to total numbers
lgz_tot.append(np.sum(large))

# Print out the stats
print("# of large sources {0}, {1}%".format(np.sum(large), percent(np.sum(large))))
print("# of non-large sources {0}, {1}%".format(np.sum(~large), percent(np.sum(~large))))

# Continue with the non-large sources...
# Check for clustering
lofar_coords = SkyCoord(mlfinal_ov["RA"], mlfinal_ov["DEC"], unit='deg', frame='icrs')

# Match with itself to find nearest neighbour separation
indx, sep2d, _ = match_coordinates_sky(lofar_coords, lofar_coords,
                                       nthneighbor=cuts["nth_nn"])

mlfinal_ov["clustered"] = np.nan
clustered = (~large) & (sep2d.arcsec <= cuts["nth_nnsep"])
# If clustered, can either be single or multiple
decision_block["clustered"] = np.sum(clustered)
decision_block["nclustered"] = np.sum(~clustered)
mlfinal_ov["clustered"][clustered] = 1.

# Print out some stats
print("# of clustered sources {0}, {1}%".format(np.sum(clustered), percent(np.sum(clustered))))
print("# of non-clustered sources {0}, {1}%".format(np.sum(~clustered), percent(np.sum(~clustered))))

# If clustered, check if there are any multiple components
mlfinal_ov["clustered_multiple"] = np.nan
mlfinal_ov["clustered_nmultiple"] = np.nan

clustered_multiple = (clustered) & (mlfinal_ov["S_Code"] != "S")
clustered_single = (clustered) & (mlfinal_ov["S_Code"] == "S")
decision_block["clustered_multiple"] = np.sum(clustered_multiple)
decision_block["clustered_nmultiple"] = np.sum(clustered_single)
mlfinal_ov["clustered_multiple"][clustered_multiple] = 1.
mlfinal_ov["clustered_nmultiple"][clustered_single] = 1.

# Print out some stats
print("# of clustered multiple sources {0}, {1}%".format(np.sum(clustered_multiple), percent(np.sum(clustered_multiple))))
print("# of clustered single sources {0}, {1}%".format(np.sum(clustered_single), percent(np.sum(clustered_single))))

# Add to total numbers
lgz_tot.append(np.sum(clustered_multiple))
prefilt_tot.append(np.sum(clustered_single))

# 
# Main blocks, if non-clustered, check if it is single or not
# 
mlfinal_ov["nclustered_single_id"] = np.nan
mlfinal_ov["nclustered_single_nid"] = np.nan

# For non-clustered single sources, check if they have a good LR
nclustered = (~large) & (sep2d.arcsec > cuts["nth_nnsep"])
nclustered_single_id = ((nclustered) & (mlfinal_ov["S_Code"] == "S") &
                        (mlfinal_ov["lr_fin"] >= lr_th))
nclustered_single_nid = ((nclustered) & (mlfinal_ov["S_Code"] == "S") &
                         (mlfinal_ov["lr_fin"] < lr_th))

mlfinal_ov["nclustered_single_id"][nclustered_single_id] = 1.
mlfinal_ov["nclustered_single_nid"][nclustered_single_nid] = 1.

# Add to total numbers
lrid_tot.append(np.sum(nclustered_single_id))
lgz_tot.append(np.sum(nclustered_single_nid))

# Print out some stats
print("# of non-clustered, single sources with ID {0}, {1}%".format(np.sum(nclustered_single_id), percent(np.sum(nclustered_single_id))))
print("# of non-clustered, single sources without ID {0}, {1}%".format(np.sum(nclustered_single_nid), percent(np.sum(nclustered_single_nid))))

# If non-single component source, check if it has a LR id
mlfinal_ov["nclustered_nsingle_id"] = np.nan
mlfinal_ov["nclustered_nsingle_nid"] = np.nan

nclustered_nsingle_id = ((nclustered) & (mlfinal_ov["S_Code"] != "S") &
                         (mlfinal_ov["lr_fin"] >= lr_th))
nclustered_nsingle_nid = ((nclustered) & (mlfinal_ov["S_Code"] != "S") &
                          (mlfinal_ov["lr_fin"] < lr_th))

mlfinal_ov["nclustered_single_id"][nclustered_single_id] = 1.
mlfinal_ov["nclustered_single_nid"][nclustered_single_nid] = 1.

# Print out some stats
print("# of non-clustered, non-single sources with ID {0}, {1}%".format(np.sum(nclustered_nsingle_id), percent(np.sum(nclustered_nsingle_id))))
print("# of non-clustered, non-single sources without ID {0}, {1}%".format(np.sum(nclustered_nsingle_nid), percent(np.sum(nclustered_nsingle_nid))))

# Add to total numbers
lrid_tot.append(np.sum(nclustered_nsingle_id))
lgz_tot.append(np.sum(nclustered_nsingle_nid))

print("\n ################### \n")
# At this stage, print out the "tentative" final sources
print("Final # of sources to send to LGZ: {0}, {1}%".format(np.sum(lgz_tot), percent(np.sum(lgz_tot))))
print("Final # of sources with good LRs: {0}, {1}%".format(np.sum(lrid_tot), percent(np.sum(lrid_tot))))
print("Final # of sources to send to Pre-filtering: {0}, {1}%".format(np.sum(prefilt_tot), percent(np.sum(prefilt_tot))))

# Question: What LR do you get if we select the LRs of the sources "tentatively" sent to LR
