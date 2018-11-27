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


def pcent_srl(number_of_sources):
    """
    Return percentage of sources w.r.t. base_sources
    """
    return number_of_sources * 100. / srl_base


def pcent_gaus(number_of_sources):
    """
    Return percentage of sources w.r.t. base_sources
    """
    return number_of_sources * 100. / gaus_base


# Read in the ML output on all overlapping area sources
# mlfinal = Table.read(path_start + "OCT17_ELAIS_im/maxl_test/full_runs/26_10_2018_2/ML_RUN_fin_overlap.fits")
mlfin_srl = Table.read("/home/rohitk/Documents/PhD/Year1/ELN1_project/OCT17_ELAIS_im/maxl_test/full_runs/26_11_2018_1/ML_RUN_fin_overlap_srl.fits")
mlfin_gaus = Table.read("/home/rohitk/Documents/PhD/Year1/ELN1_project/OCT17_ELAIS_im/maxl_test/full_runs/26_11_2018_1/ML_RUN_fin_overlap_gaul.fits")

# LR threshold (currently hard-coded)
lr_th = 0.5

# Get indices of overlapping sources in srl and gaus catalogues
indx_ov_srl, bool_ov_srl = get_overlap_sources(mlfin_srl)
indx_ov_gaus, bool_ov_gaus = get_overlap_sources(mlfin_gaus)

# Set this as the baseline of "All" sources
srl_base = len(indx_ov_srl)
gaus_base = len(indx_ov_gaus)

# Filter to get soures/Gaussians within overlapping area
mlfin_srl_ov = mlfin_srl[(indx_ov_srl)]
mlfin_gaus_ov = mlfin_gaus[(indx_ov_gaus)]

low_th = mlfin_srl_ov["lr_fin"] < lr_th

# Definition of the cuts to make
cuts = dict()
cuts["large"] = 15.         # Large size cut
cuts["bright"] = 10*1e-3    # Bright cut
cuts["nth_nn"] = 4          # n number of NNs allowed for source to be "non-clustered"
cuts["nth_nnsep"] = 45.     # nth_nn NNs within this separation allowed for source to be "non-clustered"
# cuts["nnsep"] = 45.         # NN separation (arcsec)

# Empty list to keep track of overall numbers
lgz_tot = []
lrid_tot = []
prefilt_tot = []

# Dictionary to keep track of the sources in different blocks
decision_block = dict()

# Get all the large sources
mlfin_srl_ov["large"] = np.nan
large = mlfin_srl_ov["Maj"]*3600. > cuts["large"]

decision_block["large"] = np.sum(large)
decision_block["nlarge"] = np.sum(~large)
mlfin_srl_ov["large"][large] = 1.

# Add to total numbers
lgz_tot.append(np.sum(large))

# Print out the stats
print("# of large sources {0}, {1}%".format(np.sum(large), pcent_srl(np.sum(large))))
print("# of non-large sources {0}, {1}%".format(np.sum(~large), pcent_srl(np.sum(~large))))

# Continue with the non-large sources...
# Check for clustering
lofar_coords = SkyCoord(mlfin_srl_ov["RA"], mlfin_srl_ov["DEC"], unit='deg', frame='icrs')

# Match with itself to find nearest neighbour separation
indx, sep2d, _ = match_coordinates_sky(lofar_coords, lofar_coords,
                                       nthneighbor=cuts["nth_nn"])

mlfin_srl_ov["clustered"] = np.nan
clustered = (~large) & (sep2d.arcsec <= cuts["nth_nnsep"])
# If clustered, can either be single or multiple
decision_block["clustered"] = np.sum(clustered)
decision_block["nclustered"] = np.sum(~clustered)
mlfin_srl_ov["clustered"][clustered] = 1.

# Print out some stats
print("# of clustered sources {0}, {1}%".format(np.sum(clustered), pcent_srl(np.sum(clustered))))
print("# of non-clustered sources {0}, {1}%".format(np.sum(~clustered), pcent_srl(np.sum(~clustered))))

# If clustered, check if there are any multiple components
mlfin_srl_ov["clustered_multiple"] = np.nan
mlfin_srl_ov["clustered_nmultiple"] = np.nan

clustered_multiple = (clustered) & (mlfin_srl_ov["S_Code"] != "S")
clustered_single = (clustered) & (mlfin_srl_ov["S_Code"] == "S")
decision_block["clustered_multiple"] = np.sum(clustered_multiple)
decision_block["clustered_nmultiple"] = np.sum(clustered_single)
mlfin_srl_ov["clustered_multiple"][clustered_multiple] = 1.
mlfin_srl_ov["clustered_nmultiple"][clustered_single] = 1.

# Print out some stats
print("# of clustered multiple sources {0}, {1}%".format(np.sum(clustered_multiple), pcent_srl(np.sum(clustered_multiple))))
print("# of clustered single sources {0}, {1}%".format(np.sum(clustered_single), pcent_srl(np.sum(clustered_single))))

# Add to total numbers
lgz_tot.append(np.sum(clustered_multiple))
prefilt_tot.append(np.sum(clustered_single))

# 
# Main blocks, if non-clustered, check if it is single or not
# 
mlfin_srl_ov["nclustered_single_id"] = np.nan
mlfin_srl_ov["nclustered_single_nid"] = np.nan

# For non-clustered single sources, check if they have a good LR
nclustered = (~large) & (sep2d.arcsec > cuts["nth_nnsep"])
nclustered_single_id = ((nclustered) & (mlfin_srl_ov["S_Code"] == "S") &
                        (mlfin_srl_ov["lr_fin"] >= lr_th))
nclustered_single_nid = ((nclustered) & (mlfin_srl_ov["S_Code"] == "S") &
                         (mlfin_srl_ov["lr_fin"] < lr_th))

mlfin_srl_ov["nclustered_single_id"][nclustered_single_id] = 1.
mlfin_srl_ov["nclustered_single_nid"][nclustered_single_nid] = 1.

# Add to total numbers
lrid_tot.append(np.sum(nclustered_single_id))
lgz_tot.append(np.sum(nclustered_single_nid))

# Print out some stats
print("# of non-clustered, single sources with ID {0}, {1}%".format(np.sum(nclustered_single_id), pcent_srl(np.sum(nclustered_single_id))))
print("# of non-clustered, single sources without ID {0}, {1}%".format(np.sum(nclustered_single_nid), pcent_srl(np.sum(nclustered_single_nid))))

# If non-single component source, check if it has a LR id
mlfin_srl_ov["nclustered_nsingle_id"] = np.nan
mlfin_srl_ov["nclustered_nsingle_nid"] = np.nan

nclustered_nsingle_id = ((nclustered) & (mlfin_srl_ov["S_Code"] != "S") &
                         (mlfin_srl_ov["lr_fin"] >= lr_th))
nclustered_nsingle_nid = ((nclustered) & (mlfin_srl_ov["S_Code"] != "S") &
                          (mlfin_srl_ov["lr_fin"] < lr_th))

mlfin_srl_ov["nclustered_nsingle_id"][nclustered_nsingle_id] = 1.
mlfin_srl_ov["nclustered_nsingle_nid"][nclustered_nsingle_nid] = 1.

# Print out some stats
print("# of non-clustered, non-single sources with soruce ID {0}, {1}%".format(np.sum(nclustered_nsingle_id), pcent_srl(np.sum(nclustered_nsingle_id))))
print("# of non-clustered, non-single sources without source ID {0}, {1}%".format(np.sum(nclustered_nsingle_nid), pcent_srl(np.sum(nclustered_nsingle_nid))))

# Add to total numbers
lrid_tot.append(np.sum(nclustered_nsingle_id))
lgz_tot.append(np.sum(nclustered_nsingle_nid))

"""
# Check the LR of the gaussians making up the sources
# First: Check those with a source LR > threshold
"""

source_id_m1 = mlfin_srl_ov["Source_id"][nclustered_nsingle_id]
# Get the Gaussians that make up these sources from the Gaus catalog
g_indx_m1 = np.isin(mlfin_gaus_ov["Source_id"],source_id_m1)

print("\n ################### \n")
# At this stage, print out the "tentative" final sources
print("Final # of sources to send to LGZ: {0}, {1}%".format(np.sum(lgz_tot), pcent_srl(np.sum(lgz_tot))))
print("Final # of sources with good LRs: {0}, {1}%".format(np.sum(lrid_tot), pcent_srl(np.sum(lrid_tot))))
print("Final # of sources to send to Pre-filtering: {0}, {1}%".format(np.sum(prefilt_tot), pcent_srl(np.sum(prefilt_tot))))

# Question: What LR do we get if we select the LRs of the sources "tentatively" sent to LR
