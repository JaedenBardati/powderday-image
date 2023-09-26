"""
This example code generates a filter map (filt_map.csv) that you can use to unscramble 
the Powderday band images (see Powderday-output-access.ipynb for details).

Jaeden Bardati
"""


print("Loading the modules...")

import os
from fnmatch import fnmatch

import numpy as np
import pandas as pd

import powderdayimage


powerday_output_path = '/scratch/jbardati/projectdata/powderday_output/'
filt_map_path = 'filt_map.csv'

rtout_pattern = "pd_converted_romulus_output.????_*_?.rtout.image"
hdf_pattern   = "convolved.????_*_?.hdf5"

filters = ['roman_F146', 'castor_g', 'castor_uv']
full_filter_name = lambda f: '%s_low_rez.filter' % f


print("Finding available files ...")

rtouts = {}
hdf5s  = {}

for dirpath, _, filenames in os.walk(powerday_output_path):
    for filename in filenames:
        if fnmatch(filename, rtout_pattern):
            rtouts[filename[28:-12]] = os.path.join(dirpath,filename)
        elif fnmatch(filename, hdf_pattern):
            hdf5s[filename[10:-5]] = os.path.join(dirpath,filename)

unique_halo_strs = sorted(list(set(list(rtouts.keys())).intersection(list(hdf5s.keys()))))
len_uhs = len(unique_halo_strs)


print("Making the map ...")

filt_map = pd.DataFrame(columns=['halo_str', 'old', 'new'])
filt_map['halo_str'] = list(np.repeat(unique_halo_strs, 3))
filt_map['old'] = np.nan
filt_map['new'] = np.nan

for i, halo_str in enumerate(unique_halo_strs):
    print("Halo String (%s/%s):" % (i+1, len_uhs), halo_str)

    # Manual convolving
    dc = powderdayimage.PowderdayDataCube(rtouts[halo_str], lum_unit='erg/s', wav_unit='micron', dist_unit='kpc')
    filts = [pd.read_csv('filters/%s' % full_filter_name(f), delim_whitespace=True, header=None).to_numpy() for f in filters]
    man_ims = [dc.get_convolved_image(filt) for filt in filts]
    man_sims = [(im - np.mean(im))/np.std(im) for im in man_ims]

    # Powderday convolved images
    pd_ims = [powderdayimage.SingleBandImage(hdf5s[halo_str], full_filter_name(f)).data for f in filters]
    pd_sims = [(im - np.mean(im))/np.std(im) for im in pd_ims]

    # Comparison
    similarity_matrix = np.array([[np.all(np.isclose(pd_sim, man_sim)) for man_sim in man_sims] for pd_sim in pd_sims])
    if any([sum(man_d) != 1 for man_d in similarity_matrix.T]) or any([sum(pd_d) != 1 for pd_d in similarity_matrix]): 
        raise Exception("There is not a 1-1 correspondance.\n Similarity Matrix:\n" + str(similarity_matrix))
    
    for j, pd_d in enumerate(similarity_matrix):
        filt_map.loc[i*3+j,'old'] = full_filter_name(filters[j])
        filt_map.loc[i*3+j,'new'] = full_filter_name(filters[list(pd_d).index(True)])


print("Saving the map to %s ..." % filt_map_path)
filt_map.to_csv(filt_map_path)
    