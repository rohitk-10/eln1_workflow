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

To do:
	1. How does the number of sources change when we consider 5NN - how many
	of the extra sources are compact with good LRs? If most of these are compact or of high LR then we can just change to 5NN.
	Sol: Total sent to LGZ remains ~ constant. Most of these end up having good ID so total sent to LR-ID goes from ~70% to 79%. ANd pre-filtering therefore decreases from ~20% to 11%.
	2. Deal with the ~2400 clustered_nmultiple sources. If we use 5NN, this goes down to 866.
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


def indx_to_bool(array_of_indices, array_length):
    """
    Convert an array_of_indices into a boolean array of length array_length
    """
    bool_array = np.zeros(array_length, dtype=bool)
    bool_array[array_of_indices] = True
    return bool_array


# Read in the ML output on all overlapping area sources
# mlfinal = Table.read(path_start + "OCT17_ELAIS_im/maxl_test/full_runs/26_10_2018_2/ML_RUN_fin_overlap.fits")
mlfin_srl = Table.read(path_start + "OCT17_ELAIS_im/maxl_test/full_runs/26_11_2018_1/ML_RUN_fin_overlap_srl.fits")
mlfin_gaus = Table.read(path_start + "OCT17_ELAIS_im/maxl_test/full_runs/26_11_2018_1/ML_RUN_fin_overlap_gaul.fits")

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
cuts["nth_nn"] = 5          # n number of NNs allowed for source to be "non-clustered"
cuts["nth_nnsep"] = 45.     # nth_nn NNs within this separation allowed for source to be "non-clustered"
# cuts["nnsep"] = 45.         # NN separation (arcsec)
cuts["high_lr_th"] = 10 * lr_th

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
print("# of large sources {0}, {1:3.2f}%".format(np.sum(large), pcent_srl(np.sum(large))))
print("# of non-large sources {0}, {1:3.2f}%".format(np.sum(~large), pcent_srl(np.sum(~large))))

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
print("# of clustered sources {0}, {1:3.2f}%".format(np.sum(clustered), pcent_srl(np.sum(clustered))))
print("# of non-clustered sources {0}, {1:3.2f}%".format(np.sum(~clustered), pcent_srl(np.sum(~clustered))))

# If clustered, check if there are any multiple components
mlfin_srl_ov["clustered_multiple"] = np.nan
mlfin_srl_ov["clustered_nmultiple"] = np.nan

clustered_multiple = (clustered) & (mlfin_srl_ov["S_Code"] != "S")
clustered_single = (clustered) & (mlfin_srl_ov["S_Code"] == "S")
decision_block["clustered_multiple"] = np.sum(clustered_multiple)
decision_block["clustered_nmultiple"] = np.sum(clustered_single)
mlfin_srl_ov["clustered_multiple"][clustered_multiple] = 1.
mlfin_srl_ov["clustered_nmultiple"][clustered_single] = 1.

"""
Deal with the clustered_single sources here:
	1. If 
"""

# Print out some stats
print("# of clustered multiple sources {0}, {1:3.2f}%".format(np.sum(clustered_multiple), pcent_srl(np.sum(clustered_multiple))))
print("# of clustered single sources {0}, {1:3.2f}%".format(np.sum(clustered_single), pcent_srl(np.sum(clustered_single))))

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
prefilt_tot.append(np.sum(nclustered_single_nid))

# Print out some stats
print("# of non-clustered, single sources with ID {0}, {1:3.2f}%".format(np.sum(nclustered_single_id), pcent_srl(np.sum(nclustered_single_id))))
print("# of non-clustered, single sources without ID {0}, {1:3.2f}%".format(np.sum(nclustered_single_nid), pcent_srl(np.sum(nclustered_single_nid))))

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
print("# of non-clustered, non-single sources with soruce ID {0}, {1:3.2f}%".format(np.sum(nclustered_nsingle_id), pcent_srl(np.sum(nclustered_nsingle_id))))
print("# of non-clustered, non-single sources without source ID {0}, {1:3.2f}%".format(np.sum(nclustered_nsingle_nid), pcent_srl(np.sum(nclustered_nsingle_nid))))

# Add to total numbers
lgz_tot.append(np.sum(nclustered_nsingle_nid))

"""
# M1 Branch
# Check the LR of the gaussians making up the sources
# First: Check those with a source LR > threshold
"""

"""
Three decisions:
	1. Check if there are any Gaussians with LR > threshold
		A. If none, if source LR > 10*threshold, send to LR-ID, else LGZ
		B. If all master_indices same as source-ID indices, accept LR-ID
		C. If at least one gaussian's master_index is different to source-ID index, send to LGZ
"""

all_m1_source_id = mlfin_srl_ov["Source_id"][nclustered_nsingle_id]
# Get the Gaussians that make up these sources from the Gaus catalog
all_m1_g_indx = np.in1d(mlfin_gaus_ov["Source_id"], all_m1_source_id)

# Get a list of lists with gaussians for each source in all_m1_source_id as a separate list
all_m1_grouped_g_indx = [mlfin_gaus_ov["Source_id"] == aa for aa in all_m1_source_id]

print("\n ##### In M1 branch ##### \n")
print("Total # of sources in M1 branch {0}, {1:3.2f}%".format(len(all_m1_source_id), pcent_srl(len(all_m1_source_id))))

"""
# A. If none, if source LR > 10*threshold, send to LR-ID, else LGZ
"""

# Get the source IDs from gaus catalogue with no LR threshold and those at m1 - FOR EACH SOURCE in all_m1_source_id
m1_ngid_bool = [np.all(mlfin_gaus_ov["lr_fin"][aa] < lr_th) for aa in all_m1_grouped_g_indx]
m1_ngid_source_id = all_m1_source_id[m1_ngid_bool]

# Find indices of these sources in the srl catalogue - returns a bool array
m1_ngid_srl_ov_indx = np.isin(mlfin_srl_ov["Source_id"], m1_ngid_source_id)

# Sources in m1 with no Gaus LR but with a high source lr - accept as suitable for LR
m1_ngid_hi_sid = (mlfin_srl_ov["lr_fin"] > cuts["high_lr_th"]) & (m1_ngid_srl_ov_indx)
# Sources in m1 with no Gaus LR or a high LR source id - send to LGZ
m1_ngid_hi_nsid = (mlfin_srl_ov["lr_fin"] < cuts["high_lr_th"]) & (m1_ngid_srl_ov_indx)

# Print out some stats
print("# of sources with no gaus-id BUT high source LR {0}, {1:3.2f}%".format(np.sum(m1_ngid_hi_sid), pcent_srl(np.sum(m1_ngid_hi_sid))))
print("# of sources with no gaus-id OR high source LR {0}, {1:3.2f}%".format(np.sum(m1_ngid_hi_nsid), pcent_srl(np.sum(m1_ngid_hi_nsid))))

# Add to total numbers
lrid_tot.append(np.sum(m1_ngid_hi_sid))
lgz_tot.append(np.sum(m1_ngid_hi_nsid))

"""
# B. If all master_indices same as source-ID indices, accept LR-ID
# 		AND
# C. If at least one gaussian's master_index is different to source-ID index, send to LGZ

"""

# Get the source IDs where ANY of the gaussian components have LR > threshold and those at m1
m1_gid_bool = [np.any(mlfin_gaus_ov["lr_fin"][aa] > lr_th) for aa in all_m1_grouped_g_indx]
m1_gid_source_id = all_m1_source_id[m1_gid_bool]

# Get bool array for all components of each source
m1_gid_bool_allcomp = [np.any(mlfin_gaus_ov["lr_fin"][aa] > lr_th) for aa in all_m1_grouped_g_indx]

# Get indices of these source ids into the srl catalogue
# m1_gid_srl_ov_indx = np.isin(mlfin_srl_ov["Source_id"], m1_gid_source_id)

# Get LRs of these sources from the srl and gaus catalogue (this will have duplicate source_ids, on purpose)
m1_gid_srl_lr_index = np.array(mlfin_srl_ov["lr_index_fin"][np.searchsorted(mlfin_srl_ov["Source_id"], m1_gid_source_id)])

all_m1_gaus_lr_index = [mlfin_gaus_ov["lr_index_fin"][aa] for aa in all_m1_grouped_g_indx]
all_m1_gaus_lr = [mlfin_gaus_ov["lr_fin"][aa] for aa in all_m1_grouped_g_indx]
# m1_gid_gaus_lr_bool = [np.any(aa > lr_th) for aa in all_m1_gaus_lr]
m1_gid_gaus_lr_index = np.array(all_m1_gaus_lr_index)[m1_gid_bool]


# Simply take the difference of the two and if the absolute value is greater than 0.5, then srl and its gaus component are not matching to the same optical source!
same_lr_index = np.array([np.all(cc < 0.5) for cc in np.abs(m1_gid_srl_lr_index - m1_gid_gaus_lr_index)])
diff_lr_index = ~same_lr_index

# Done need to take np.unique of the second arguments in np.isin?
diff_lrindex_srl_indx = np.isin(mlfin_srl_ov["Source_id"], m1_gid_source_id[diff_lr_index])
same_lrindex_srl_indx = np.isin(mlfin_srl_ov["Source_id"], m1_gid_source_id[same_lr_index])

# Add to total numbers
lrid_tot.append(np.sum(same_lr_index))
lgz_tot.append(np.sum(diff_lr_index))

print("# Braches M1 - B and C #")
print("# of sources with all gaus-id index same as source-id index {0}, {1:3.2f}%".format(np.sum(same_lrindex_srl_indx), pcent_srl(np.sum(same_lrindex_srl_indx))))
print("# of sources with >= 1 different gaus-id index to source-id index high source LR {0}, {1:3.2f}%".format(np.sum(diff_lrindex_srl_indx), pcent_srl(np.sum(diff_lrindex_srl_indx))))

print("\n ##### End of M1 branch #####")
print("\n ##### In M2 branch #####")

"""
# M2 Branch
"""

all_m2_source_id = mlfin_srl_ov["Source_id"][nclustered_nsingle_nid]
# Get the Gaussians that make up these sources from the Gaus catalog
all_m1_g_indx = np.in1d(mlfin_gaus_ov["Source_id"], all_m1_source_id)

# Get a list of lists with gaussians for each source in all_m1_source_id as a separate list
all_m1_grouped_g_indx = [mlfin_gaus_ov["Source_id"] == aa for aa in all_m1_source_id]

print("\n ##### In M1 branch ##### \n")
print("Total # of sources in M1 branch {0}, {1:3.2f}%".format(len(all_m1_source_id), pcent_srl(len(all_m1_source_id))))


# Print out final values in each of the end-points
print("\n ################### \n")
# At this stage, print out the "tentative" final sources
print("Final # of sources to send to LGZ: {0}, {1:3.2f}%".format(np.sum(lgz_tot), pcent_srl(np.sum(lgz_tot))))
print("Final # of sources with good LRs: {0}, {1:3.2f}%".format(np.sum(lrid_tot), pcent_srl(np.sum(lrid_tot))))
print("Final # of sources to send to Pre-filtering: {0}, {1:3.2f}%".format(np.sum(prefilt_tot), pcent_srl(np.sum(prefilt_tot))))


end_point_sum = np.sum(lgz_tot) + np.sum(lrid_tot) + np.sum(prefilt_tot)
print("Total number of sources in all end-points: {0}, {1:3.2f}%".format(end_point_sum, pcent_srl(end_point_sum)))

assert end_point_sum == srl_base, "Number of sources in end points don't match up with total number of sources"

# Question: What LR do we get if we select the LRs of the sources "tentatively" sent to LR
