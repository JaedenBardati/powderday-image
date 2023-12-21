#!/usr/bin/env python
# coding: utf-8

# # GENERATE MORPHS

# This code generates the morphological properties from the available image data and stores it in two files called `event_morphs.csv` and `control_morphs.csv`. They contain all the data needed for morph analysis. Note that the `event_morphs.csv` dataframe rows are not unique to a halo image (since there can be multiple BH merger events in an image), whereas the `control_morphs.csv` is.
# 
# Unless you are testing or running on a sample sample, **please use the python file version instead** to avoid a memory leaking issue which only occurs in Jupyter Notebook.

# ## Import libraries

# In[1]:


RUNIDEN = '_JUN5'


# Imports
import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import matplotlib.cm as cm

import scipy
import scipy.stats
import scipy.ndimage as ndi
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy import special
from scipy import optimize

import astropy
from astropy.visualization import simple_norm
from astropy.modeling import models
from astropy import convolution
from astropy.convolution import convolve

import pandas as pd

from IPython.display import display, HTML

import photutils
from photutils.segmentation import detect_threshold as photutils_detect_threshold
from photutils.segmentation import detect_sources as photutils_detect_sources
from photutils.segmentation import deblend_sources as photutils_deblend_sources


import statmorph
from statmorph.utils.image_diagnostics import make_figure

import h5py

import time
import os
import os.path
import sys

import gc
import weakref
import tracemalloc


# ## Load the image metadata

# ### Functions and classes to load images

# In[3]:


# some functions/classes
def percentile(data, percent, reverse_order=False):
    return sorted(data, reverse=reverse_order)[int(math.ceil((len(data) * percent) / 100)) - 1]

def normalize_filter(image_data, bot=5, top=95, dynamic_range=None, reverse_order=False, ret_minmax=False, asinh_fit=False):
    if reverse_order: image_data = -image_data
    max_=percentile(image_data.ravel(), top, reverse_order=False) if top != 100 else np.max(image_data.ravel())
    
    if dynamic_range is None:
        min_=percentile(image_data.ravel(), bot, reverse_order=False) if bot != 0 else np.min(image_data.ravel())
    else:
        bot = 0
        min_ = max_ - dynamic_range
        
    image_data = (image_data - min_) / (max_ - min_)
    
    if asinh_fit: image_data = np.arcsinh(image_data*asinh_fit)

    image_data = (image_data*(top - bot) + bot)/100
    
    image_data[image_data > 1] = 1
    image_data[image_data < 0] = 0
    
    if ret_minmax:
        return image_data, min_, max_
    return image_data

def trapezoidal_integration(xs, ys):
    return ((ys[:-1]+np.asarray(ys)[1:])*np.diff(xs)).sum()/2.

def sqdist_to_center_array(size, center=None):
    if center is None: 
        center = (np.array([sx, sy], dtype=float) - 1)/2.
    else:
        center = np.array(center, dtype=float)
        center[1] = size[1] - center[1]-1
        
    y, x = np.ogrid[:size[1], :size[0]]
    return (y - center[1]) ** 2 + (x - center[0]) ** 2


class SpectralImage:
    """A class to wrap the image data from Powderday."""
    
    @staticmethod
    def _load_image(filepath, log=False):
        """
        Loads an hdf5 image file at filepath (.hdf5), logs some properties if desired, and returns the resulting object.
        """
        if log: 
            print("Loading hdf5 file at {} . . .".format(filepath))
        
        f = h5py.File(filepath, 'r')
        
        if log: 
            print("The filters are: {}".format([filt.decode("utf-8", "ignore") for filt in f['filter_names']]))
            print()

        return f
    
    @staticmethod
    def _zoom_image(array2d, zoom_factor):
        if zoom_factor < 1:
            raise Exception("The zoom factor must be greater than or equal to 1.")
        xsize, ysize = array2d.shape[0], array2d.shape[0]  # size (in number of pixel space)
        cx, cy = int(xsize/2.), int(ysize/2.) # center pixel
        xnhs, ynhs = int(xsize/(zoom_factor*2.)), int(xsize/(zoom_factor*2.))  # new half size
        return array2d[cx-xnhs:cx+xnhs, cy-ynhs:cy+ynhs]
    
    @staticmethod
    def _rotate_cc90_2d(array2d):
        return np.flip(array2d.transpose(1, 0), axis=0)
    
    @staticmethod
    def _rotate_cc90_3d(array3d): # rotation keeping axis 0 constant
        return np.flip(array3d.transpose(0, 2, 1), axis=2)
    
    @staticmethod
    def finish_up_plot(output_filepath=None, log=False, noshow=False):
        if log: print("Saving figure . . .")
        if output_filepath is not None:
            plt.savefig(output_filepath, bbox_inches='tight', edgecolor='white', transparent=False)
        elif not noshow: 
            plt.show()
        if log: print()
        
    def _update_npix(self, shape): 
        """Updates npix and w given the (2d) shape of the data"""
        if np.isscalar(shape) or len(shape) != 2 or shape[0] != shape[1]:
            raise NotImplementedError("Only 2d square images are supported.")
        self.npix = shape[1]
        self.datashape = np.array([self.npix, self.npix])    # shape of the pixel data (for a single filter)
        self.pixcent = (self.datashape - 1)/2.
    
    def close(self):
        self._f.close()
        del self._f
    
    def __init__(self, filepath, log=False, closef=True, rotate_cc90=True): # rotate_cc90 required for powderday images only
        self._filepath = filepath
        self._f = self._load_image(self._filepath, log=log)
        
        self.filters = [filt_.decode("utf-8", "ignore") for filt_ in self._f['filter_names']]  #.astype(str)
        self.image_data = np.array(self._f['image_data'])
        
        self.rotate_cc90 = rotate_cc90
        if rotate_cc90: self.image_data = self._rotate_cc90_3d(self.image_data)
        
        self.w = self._f['image_data'].attrs['width']
        self.w_unit = self._f['image_data'].attrs['width_unit'].astype(str) if type(self._f['image_data'].attrs['width_unit']) is not str else self._f['image_data'].attrs['width_unit']
        
        self._update_npix(self.image_data.shape[1:])
        self.pixsize = 2*self.w/self.npix    # in units of [self.w_unit]
        if closef: self.close()
        
    def _get_filter_data(self, filt):
        if filt not in self.filters:
            raise Exception("The filter '{}' is not in the dataset.\nAvailable filters are: {}".format(filt, self.filters))
        return self.image_data[self.filters.index(filt)]
    
    def filter_image(self, filt, zoom_factor=None, mask=None, flip_axis=None, log=False):
        """Returns a MonochromaticImage object of a given filter."""
        return MonochromaticImage(self._filepath, filt, rotate_cc90=self.rotate_cc90, zoom_factor=zoom_factor, mask=mask, flip_axis=flip_axis, log=log)
    
    def plot_rgb(self, rfilt, gfilt, bfilt, output_filepath=None, m=0, alpha=0.02, Q=8, weights=[1,1,1], scales=[1, 1, 1],psf_sigma=None, 
                 xlim=None, ylim=None, title=None, zoom_factor=1, log=False, debug=False, simplergb=False, logplot=True, no_normalize=False,
                 pynbody_mode=False,dynamic_range=5.0, fontsize=None, labelsize=None, return_image=False, imagepremodfoo=lambda image: image):
        """
        Plots the RGB image using 3 given filters. 
        If output_filepath is not None, it saves the result at that location (.png).
        """
        
        if log: print("Generating image . . .")
        R = imagepremodfoo(self._get_filter_data(rfilt)) * scales[0]
        G = imagepremodfoo(self._get_filter_data(gfilt)) * scales[1]
        B = imagepremodfoo(self._get_filter_data(bfilt)) * scales[2]

        if pynbody_mode:
            def combine(r, g, b, magnitude_range, brightest_mag=None, mollview=False):
                # flip sign so that brightest pixels have biggest value
                r = -r
                g = -g
                b = -b

                if brightest_mag is None:
                    brightest_mag = []

                    # find something close to the maximum that is not quite the maximum
                    for x in r, g, b:
                        if mollview:
                            x_tmp = x.flatten()[x.flatten()<0]
                            ordered = np.sort(x_tmp.data)
                        else:   
                            ordered = np.sort(x.flatten())
                        brightest_mag.append(ordered[-len(ordered) // 5000])

                    brightest_mag = max(brightest_mag)
                else:
                    brightest_mag = -brightest_mag

                rgbim = np.zeros((r.shape[0], r.shape[1], 3))
                rgbim[:, :, 0] = bytscl(r, brightest_mag - magnitude_range, brightest_mag)
                rgbim[:, :, 1] = bytscl(g, brightest_mag - magnitude_range, brightest_mag)
                rgbim[:, :, 2] = bytscl(b, brightest_mag - magnitude_range, brightest_mag)
                return rgbim, -brightest_mag

            def bytscl(arr, mini=0, maxi=10000):
                X = (arr - mini) / (maxi - mini)
                X[X > 1] = 1
                X[X < 0] = 0
                return X

            def convert_to_mag_arcsec2(image):
                pc2_to_sqarcsec = 2.3504430539466191e-09
                return -2.5*np.log10(image*pc2_to_sqarcsec)

            def nw_scale_rgb(r, g, b, scales=[4, 3.2, 3.4]):
                return r * scales[0], g * scales[1], b * scales[2]

            def nw_arcsinh_fit(r, g, b, nonlinearity=3):
                radius = r + g + b
                val = np.arcsinh(radius * nonlinearity) / nonlinearity / radius
                return r * val, g * val, b * val

            def pynbody_render(r, g, b, dynamic_range=5.0):
                r=convert_to_mag_arcsec2(r)
                g=convert_to_mag_arcsec2(g)
                b=convert_to_mag_arcsec2(b)

                #r,g,b = nw_scale_rgb(r,g,b)
                #r,g,b = nw_arcsinh_fit(r,g,b)

                rgbim, mag_max = combine(r, g, b, dynamic_range*2.5)
                mag_min = mag_max + 2.5*dynamic_range

                return rgbim

            rgb_image = pynbody_render(R, G, B, dynamic_range=dynamic_range)
        else:
            if psf_sigma is not None:
                psf_size = self.npix//2-1  # on each side from the center
                y, x = np.mgrid[-psf_size:psf_size+1, -psf_size:psf_size+1]
                psf = np.exp(-(x**2 + y**2)/(2.0*psf_sigma**2))
                psf /= np.sum(psf)
                R = convolve(R, psf)
                G = convolve(G, psf)
                B = convolve(B, psf)

            if logplot:
                R = np.log(R)
                G = np.log(G)
                B = np.log(B)

            R = self._zoom_image(R, zoom_factor)
            G = self._zoom_image(G, zoom_factor)
            B = self._zoom_image(B, zoom_factor)

            if not no_normalize:
                max_, min_ = max([np.max(R), np.max(G), np.max(B)]), min([np.min(R), np.min(G), np.min(B)])
                R = (R - min_) / (max_ - min_)   # normalize
                G = (G - min_) / (max_ - min_)
                B = (B - min_) / (max_ - min_)

            if not simplergb:
                M = m + np.sinh(Q)/(alpha*Q)
                F = lambda x: np.arcsinh(alpha*Q*(x - m))
                func = lambda x: 0 if x < m else F(x-m)/F(M-m) if M >= x else 1

                I = (R + G + B)/3
                if debug: print(I, m, M)

                # weigh and normalize by new min
                factor = np.array([[func(i) for i in ii] for ii in I]) / I
                np.nan_to_num(factor, copy=False, nan=0.0)
                if debug: print("factors: ", factor)
                R = R**weights[0] * factor 
                G = G**weights[1] * factor
                B = B**weights[2] * factor

                max_RGB = np.max([R, G, B])
                R /= max_RGB
                G /= max_RGB
                B /= max_RGB

            rgb_image = np.dstack((R, G, B))
            
        if return_image:
            return rgb_image

        if log: print("Plotting image . . .")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        neww = self.w if zoom_factor is None else self.w * (R.shape[1]/self.npix)
        cax = ax.imshow(rgb_image, origin='lower', extent=[-neww, neww, -neww, neww])
        ax.set_xlabel('x ({})'.format(self.w_unit), fontsize=fontsize)
        ax.set_ylabel('y ({})'.format(self.w_unit), fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=labelsize)
        ax.tick_params(axis='y', labelsize=labelsize)

        if title is None:
            title = "RGB image"
        plt.title(title)
        plt.tight_layout()

        plt.xlim(xlim)
        plt.ylim(ylim)  

        fig.patch.set_facecolor('white')
        self.finish_up_plot(output_filepath=output_filepath, log=log)

    
class MonochromaticImage(SpectralImage):
    """A class that inherits from the SpectralImage class."""
    
    def __init__(self, filepath, filt, rotate_cc90=True, zoom_factor=None, mask=None, flip_axis=None, log=False):  # rotate_cc90 required for powderday images only
        super().__init__(filepath, log=log, rotate_cc90=rotate_cc90)
        self._zoom_factor, self._mask = zoom_factor, mask
        
        self.filter_name = filt
        self.data = self._get_filter_data(filt) # must have a filter
        
        if flip_axis == 0 or flip_axis == 1: 
            self.data = np.flip(self.data, axis=flip_axis)
        if flip_axis == 2: 
            self.data = np.flip(np.flip(self.data, axis=0), axis=1)
        
        if self._zoom_factor is not None: 
            prenpix = self.npix
            
            self.data = self._zoom_image(self.data, self._zoom_factor)
            self._update_npix(self.data.shape)
            if log: print("New data size: %d x %d" % tuple(self.datashape))
            
            self._real_zoom_factor = prenpix/self.npix
            self.w /= self._real_zoom_factor
            if log: print("Real zoom factor:", self._real_zoom_factor)
        
        
    # Plotting routines
    def plot_image(self, in_magnitude=False, xlim=None, ylim=None, title=None, cmap=plt.cm.viridis, 
                   output_filepath=None, log=False, showstuff=False, zero_to_min=False, tight=False, 
                   vmin=None, finish_plot=True):
        """
        Plots the monochromatic image. 
        If output_filepath is not None, it saves the result at that location (.png).
        """
        if log: print("Plotting image . . .")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        showndata = self.data # in ergs/s
        
        if zero_to_min: showndata[showndata == 0] = np.min(showndata[showndata != 0])
            
        if vmin is not None: showndata[showndata < vmin] = vmin
        
        if in_magnitude:
            showndata = -2.5*np.log10(showndata/3.0128e35) # ergs/s
            cmap = cmap.reversed()
        
        cax = ax.imshow(showndata, cmap=cmap, origin='lower', extent=[-self.w, self.w, -self.w, self.w])
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel('x ({})'.format(self.w_unit))
        ax.set_ylabel('y ({})'.format(self.w_unit))
        if in_magnitude:
            plt.colorbar(cax, label='Magnitude')
        else:
            plt.colorbar(cax, label='Luminosity (ergs/s)', format='%.0e')

        if title is None:
            title = "Monochromatic Image for filter: {}".format(self.filter_name)
        plt.title(title)
        if tight: plt.tight_layout()

        plt.xlim(xlim)
        plt.ylim(ylim)
        
        if showstuff:
            _c = self.centroid()
            plt.scatter(*_c, color='red', marker='+')
            ax.add_patch(plt.Circle(_c, self.r20(center=_c), color='orange', fill=False, lw=2, ls="--"))
            ax.add_patch(plt.Circle(_c, self.r80(center=_c), color='yellow', fill=False, lw=2, ls="--"))
            plt.legend(["Centroid", "R20", "R80"])

        fig.patch.set_facecolor('white')
        if finish_plot: 
            self.finish_up_plot(output_filepath=output_filepath, log=log)
        else:
            return fig, ax


# ### Obtain the metadata

# In[4]:


SAMPLE_METADATA_FILE = "Image_Data/sample_events.metadata"
CONTROL_METADATA_FILE = "Image_Data/control.metadata"

def load_custom_pd(file_path, cols_to_eval=[], parallel=True, use_custom_na=True, na_rep='NA',
                   columns=None, header='infer', usecols=None, index_col=0, verbose=False):
    """Loads custom file using pandas. Uses the list <columns> as the column names. If <header> is 
        'infer', it will use what the file reads to be the header as the name of the columns. If <header> 
        is instead None, it will not do so. <index_col> = None for no inedx reading."""
    if use_custom_na:
        df = pd.read_csv(file_path, delim_whitespace=True, header=header, usecols=usecols, index_col=index_col, na_values=na_rep, keep_default_na=False, na_filter=True) # Load in data and name columns
    else:
        df = pd.read_csv(file_path, delim_whitespace=True, header=header, usecols=usecols, index_col=index_col, keep_default_na=True, na_filter=True) # Load in data and name columns

    if header is not None and header != 'infer': df.columns = columns
    def eval_cols(x):
        return x if pd.isna(x) else ast.literal_eval(x)
    for col in cols_to_eval: df[col] = df[col].parallel_apply(eval_cols)

    return df

def save_custom_pd(df, file_path, sep=' ', na_rep='NA', header=True, index=False):
    """Saves custom file using pandas. If <header> is True, it will write columns as the first line, but 
    <header> is False, it will not. If <index> is True, it will write the row names, if False, it will not."""
    df.to_csv(file_path, sep=sep, header=header, index=index, na_rep=na_rep)


def readDSVfile(fn, startl=None, endl=None, delim=None, get_cols=False):
    '''
    Reads a delimiter-separated data file as a multidimensional numpy array of rows (of strings). 
    
    Parameters:
     ** fn                       : Filename to extract data from.
        startl    (default=None) : Line to start reading from inclusively (zero-indexed). By default starts on line 0.
        endl      (default=None) : Line to end reading from inclusively (zero-indexed). By default, ends on last line.
        delim     (default=None) : The separator used in the data.
        get_cols (default=False) : Flag to return a list of columns instead (i.e. in the form [col][row]).
    '''
    
    # Defaults
    if startl is None: startl = 0
    if endl is None: endl = np.inf
    if delim is None: delim = ' '
    
    # Open file
    with open(fn) as f:
        # Get data between rows starl and endl inclusively
        data = np.array([line.strip().split(delim) for i, line in enumerate(f) if i >= startl and i <= endl])
    
    # return 1d array if ...
    if data.shape[0] == 1: data = data.reshape(data.shape[1])    # only 1 row of data
    elif data.shape[1] == 1: data = data.reshape(data.shape[0])  # only 1 col of data
    
    # adjust array if [col][row] form is desired
    if get_cols: data = np.transpose(data)
    
    # Return data
    return data


# In[5]:


# temp: investigating on march 14 of 2023
events_df = load_custom_pd(SAMPLE_METADATA_FILE, index_col=None)
events_df


# In[6]:


events_df[events_df['event id'] == 5652].drop_duplicates(subset=["host halonumber", "available step"], keep='first')[['event id', 'available step', 'host halonumber', 'Mstar', 'survived id', 'eaten id']]


# In[7]:


list(events_df.columns)


# In[8]:


control_df = load_custom_pd(CONTROL_METADATA_FILE, index_col=None)
control_df


# In[9]:


# Obtain sample and control list files and ensure they are the matched to the events_df and control_df

sample_halos = readDSVfile("Image_Data/sample_halos.txt", delim=',')  # should be the same as events_df but has unique entries
control_halos = readDSVfile("Image_Data/pop_sample_halos.txt", delim=',') # should have unique entries like control_df


if len(set(["_".join(l) for l in sample_halos])) != len(["_".join(l) for l in sample_halos]):
    raise Exception("Sample halos are not unique!")
        
if len(set(["_".join(l) for l in control_halos])) != len(["_".join(l) for l in control_halos]):
    raise Exception("Control halos are not unique!")

if (set(list(zip(events_df['host halonumber'], events_df['available step']))) != 
    set([(int(hn), int(step)) for hn, step in sample_halos])): 
    raise Exception("Sample halos not matching. You likely need to update sample_events.metadata or sample_halos.txt")

if (set(list(zip(control_df['host halonumber'], control_df['available step']))) != 
    set([(int(hn), int(step)) for hn, step in control_halos])): 
    raise Exception("Control halos not matching. You likely need to update control.metadata or pop_sample_halos.txt")


# ### Define our master dataframes

# In[10]:


# Define the inclinations available
inclinations = [0, 1, 2, 3]

trans_mats = [  # the index is the vertex
    np.array([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]),
    np.array([[1., 0., 0.],[0., -0.33333333, -0.94280904],[0.,  0.94280904, -0.33333333]]),
    np.array([[-0.33333333, -0.94280904, 0.],[-0.31426968, 0.11111111, -0.94280904],[0.88888889, -0.31426968, -0.33333333]]),
    np.array([[-0.33333333, 0.94280904, 0.],[ 0.31426968, 0.11111111, -0.94280904],[-0.88888889, -0.31426968, -0.33333333]])
] # NOTE THAT THE ABOVE MAY NEED TO BE REWORKED IN THE FUTURE WHEN DEALING WITH RERUNS (with perturbed angles)


# Define the filters used for the convolved.* files (powderday)
filters_conv = ['roman_F146_low_rez.filter','castor_g_low_rez.filter','castor_uv_low_rez.filter'] 


# Create master dataframes
master_events = events_df.copy()
master_control = control_df.copy()


# add major-minor qualifiers for events
master_events['major'] = master_events['mass ratio'] >= 0.25


# add orientations
edfs, cdfs = [], []
for inc in inclinations:
    subdf = master_events.assign(orientation=(np.ones(len(master_events))*inc).astype(int))
    subdf[['central offset x', 'central offset y', 'central offset z']] = np.dot(   # tranform bh position
        subdf[['central offset x', 'central offset y', 'central offset z']],
        trans_mats[inc]
    )
    edfs.append(subdf)
    cdfs.append(master_control.assign(orientation=(np.ones(len(master_control))*inc).astype(int)))
    
master_events=pd.concat(edfs)
master_control=pd.concat(cdfs)


#add halo_str
master_events['halo_str'] = (master_events['available step'].astype(int).astype(str) + '_' + 
                             master_events['host halonumber'].astype(int).astype(str) + '_' + 
                             master_events['orientation'].astype(int).astype(str))

master_control['halo_str'] = (master_control['available step'].astype(int).astype(str) + '_' + 
                             master_control['host halonumber'].astype(int).astype(str) + '_' + 
                             master_control['orientation'].astype(int).astype(str))


# check if its image file is available
master_events['image exists'] = master_events.apply(
    lambda row: os.path.exists("Image_Data/convolved.{}.hdf5".format(row['halo_str'])), 
    axis=1
)

master_control['image exists'] = master_control.apply(
    lambda row: os.path.exists("Image_Data/convolved.{}.hdf5".format(row['halo_str'])), 
    axis=1
)

_ne = len(master_events[master_events['image exists'] == True])
_nc = len(master_control[master_control['image exists'] == True])
print("Only %.2f%% of images found (sample=%.1f%%, control=%.1f%%)." % (
    100*(_nc+_ne)/(len(master_events)+len(master_control)),
    100*_ne/len(master_events),
    100*_nc/len(master_control)
))


# sanity check
assert len(set(list(master_events['halo_str']))) == len(sample_halos)*len(inclinations)
assert len(master_events[master_events['mass ratio'] < 0.25]) == len(master_events[master_events['major'] == False])


# drop all unavailable images
_ple, _plc = len(master_events), len(master_control)

master_events=master_events.reset_index(drop=True)
master_events.drop(master_events[master_events['image exists'] != True].index, inplace=True)
master_events.drop(columns=['image exists'], inplace=True)

master_control=master_control.reset_index(drop=True)
master_control.drop(master_control[master_control['image exists'] != True].index, inplace=True)
master_control.drop(columns=['image exists'], inplace=True)

print("Dropped the missing %.2f%% (sample=%.1f%%, control=%.1f%%)." % (
    100*(1-(len(master_events)+len(master_control))/(_ple+_plc)),
    100*(1-len(master_events)/_ple),
    100*(1-len(master_control)/_plc),
))


# add filters
master_events=pd.concat([master_events.assign(filter=[filt for _ in range(len(master_events))]) for filt in filters_conv])
master_control=pd.concat([master_control.assign(filter=[filt for _ in range(len(master_control))]) for filt in filters_conv])


# sort by identifiers 
master_events=master_events.sort_values(by=['available step', 'host halonumber', 'orientation', 'filter', 'delay time'])
master_events=master_events.reset_index(drop=True)
master_events=master_events.astype({**{c:float for c in master_events.columns}, **{"available step":int, "host halonumber":int, "halo_str":str, "filter":str, "event id":int, "post bhid":int, "major": bool, "orientation": int}})


master_control=master_control.sort_values(by=['available step', 'host halonumber', 'orientation', 'filter'])
master_control=master_control.reset_index(drop=True)
master_control=master_control.astype({**{c:float for c in master_control.columns},**{"available step":int, "host halonumber":int, "halo_str":str, "filter":str, "orientation": int}})

# display
print()
print("EVENTS:")
display(master_events)
print()
print("CONTROL:")
display(master_control)


# ### Get the morphological measurements

# In[29]:


DEFAULT_SEGOPTIONS = dict(
    add_noise=100,              # S/N at Reff
    discretize=False,           # convert to discrete counts
    psf_sigma=1/2.355,          # fwhm of 1 pixel
    npixels=10,                 # minimum number of connected pixels
    seg_thres=10,      ##10 
    dblnd_nlevels=32,  ##8 
    dblnd_contrast=0.1,##0.001  
    dblnd_kernel_sigma=4,       # N pixel sigma
    dblnd_kernel_size=None,     # kernel size of MxM pixels
    regularize=5,               # regularization boxcar size 
)

DEFAULT_MORPH_OPTIONS = dict(
    gain=1,   # leave this as 1: like this, it is (s/n)**2 over the flux average at Reff
    verbose=False,
    petro_extent_cas=1.5,      ## 2        # default: 1.5
    petro_fraction_cas=0.25,   ##0.1       # default: petro_fraction_cas=0.25
    petro_fraction_gini=0.2,   ##0.5       # default: petro_fraction_gini=0.2 
    skybox_size=32,                        # default: 32
    cutout_extent=1.5,                     # default: 1.5
    boxcar_size_mid=3.0,        ##0.0      # default: 3.0
    boxcar_size_shape_asym=3.0, ##2.0      # default: 3.0
    petro_extent_flux=2.0,  ## may+:0.5    # default: 2.0 
)



def find_bhmergerpos(halo_str, subim, in_pix=True, flip_y=False, verbose=False):
    """find the position of the COM of the available merging BHs for the halo"""
    matched_halos = master_events[master_events['halo_str'] == str(halo_str)]
    if matched_halos.empty:
        if verbose: print("WARNING: BH data not available! This may be a control sample halo.")
        return None

    matched_halo = matched_halos.iloc[0] ## THIS MUST BE CHECKED!
    
    # get bh pos
    bhmergerpos = [matched_halo['central offset x'], matched_halo['central offset y'], matched_halo['central offset z']]
    
    if in_pix:
        bhmergerpos = np.round(subim.npix*(bhmergerpos+subim.w)/(2*subim.w)).astype(int)   # convert to pixels
        if flip_y: bhmergerpos[1] = subim.npix - bhmergerpos[1]  # fix powderday y-axis flip 
        return bhmergerpos
    
    return bhmergerpos


def _fraction_of_total_function_circ(r, image, center, fraction, total_sum):
    """
    Helper function to calculate ``_radius_at_fraction_of_total_circ``.
    """
    assert (r >= 0) & (fraction >= 0) & (fraction <= 1) & (total_sum > 0)
    if r == 0:
        cur_fraction = 0.0
    else:
        ap = photutils.aperture.CircularAperture(center, r)
        # Force flux sum to be positive:
        ap_sum = np.abs(ap.do_photometry(image, method='exact')[0][0])
        cur_fraction = ap_sum / total_sum

    return cur_fraction - fraction

def _radius_at_fraction_of_total_circ(image, center, r_total, fraction):
    """
    Return the radius (in pixels) of a concentric circle that
    contains a given fraction of the light within ``r_total``.
    """
    flag = 0

    ap_total = photutils.aperture.CircularAperture(center, r_total)

    total_sum = ap_total.do_photometry(image, method='exact')[0][0]
    assert total_sum != 0
    if total_sum < 0:
        warnings.warn('[r_circ] Total flux sum is negative.', AstropyUserWarning)
        flag = 2
        total_sum = np.abs(total_sum)

    # Find appropriate range for root finder
    npoints = 100
    r_grid = np.linspace(0.0, r_total, num=npoints)
    i = 0  # initial value
    while True:
        assert i < npoints, 'Root not found within range.'
        r = r_grid[i]
        curval = _fraction_of_total_function_circ(
            r, image, center, fraction, total_sum)
        if curval <= 0:
            r_min = r
        elif curval > 0:
            r_max = r
            break
        i += 1

    r = scipy.optimize.brentq(_fraction_of_total_function_circ, r_min, r_max,
                   args=(image, center, fraction, total_sum), xtol=1e-6)

    return r, flag


def get_morphs(halo_str, filter_, 
              imagetype='convolved', subdir='', zoom_factor=None,
              deblend=True, background_subtract=False, 
              check_black_holes=True, assert_check_black_holes=True, max_width=90,
              nodisplay=False, extraprints=True, outputfiles=None, two_morphs=False,
              segoptions=DEFAULT_SEGOPTIONS,morphoptions=DEFAULT_MORPH_OPTIONS):
    start = time.time()
    
    if outputfiles is None: outputfiles = [None, None, None]
    
    ## Acceptable image types: 'convolved' or 'render'
    try:
        subim = MonochromaticImage('Image_Data/{}/{}.{}.hdf5'.format(subdir, imagetype, halo_str), filter_, zoom_factor=zoom_factor, log=extraprints)
    except OSError:
        return Exception("error-code-4") 
    
    print("Halo String: {}     Filter: {}".format(halo_str, filter_))
    image = subim.data
    #if imagetype=='render': image = 10**(image*8)
    #if image_thres is not None: image[image < image_thres] = 0 #image_thres
    if extraprints: print("Image shape:", image.shape)
    
    # --- Get BHs ---
    bhmergerpos_inpix = None if not check_black_holes else find_bhmergerpos(halo_str, subim, verbose=extraprints, in_pix=True, flip_y=True)
    
    # --- begin plot ---
    if not nodisplay:
        figshape = np.array([2, 3 if deblend else 2], dtype=int)
        fig, axes = plt.subplots(figshape[1], figshape[0], sharey=True, sharex=True, figsize=tuple(figshape*5))
        if segoptions["psf_sigma"] is None:
            axes[0, 0].imshow(image, origin='lower', cmap='gray')
            axes[0, 0].set_title('Image (flux)'.format())
        if bhmergerpos_inpix is not None:
            axes[0, 1].scatter(bhmergerpos_inpix[0], bhmergerpos_inpix[1], marker='x',color='red', label="BH")
            axes[0, 1].legend()
        
        # -temp-
#         circle1 = plt.Circle((50, 50), np.sqrt((50-bhmergerpos_inpix[0])**2+(50-bhmergerpos_inpix[1])**2), color='r', fill=False)
#         axes[0, 1].add_patch(circle1)
        # --temp --
#         _bhs = [k for k in sample_properties[halo_str].keys() if k[:13] == 'BH_transform_']
#         if _bhs != []:
#             _bhposes = [sample_properties[halo_str][k]['BH_central_offset'] for k in _bhs if 'BH_central_offset' in sample_properties[halo_str][k].keys() and sample_properties[halo_str][k]['BH_mass'] >= 1e7]
#             _bhposes = [np.round(subim.npix*(np.array(bhp)+subim.w)/(2*subim.w)).astype(int) for bhp in _bhposes]
#             _bhposes = [bh for bh in _bhposes if 0 < bh[0] < subim.npix and 0 < bh[1] < subim.npix]
#             axes[0, 1].scatter([bh[0]for bh in _bhposes], [bh[1]for bh in _bhposes], color='blue', marker='.')
#         # ---- 
    
    # --- convolve with PSF ---
    if segoptions["psf_sigma"] is not None:
        psf_size = subim.npix//2-1  # on each side from the center
        y, x = np.mgrid[-psf_size:psf_size+1, -psf_size:psf_size+1]
        psf = np.exp(-(x**2 + y**2)/(2.0*segoptions["psf_sigma"]**2))
        psf /= np.sum(psf)
        if not nodisplay:
            axes[0, 0].imshow(psf, origin='lower', cmap='gray')
            axes[0, 0].set_title('PSF')
            axes[0, 0].patch.set_facecolor('black')
        image = convolve(image, psf)
    
    # --- add noise ---
    if segoptions["add_noise"] is not None:
        #np.random.seed(1)
        
        # calculate image centroid
        try:
            _xs = np.tile(np.arange(0, image.shape[0]),(image.shape[1],1))
            _ys = np.tile(np.arange(0, image.shape[1]).reshape((image.shape[1], 1)), image.shape[0])

            cx = (_xs*image).sum()/image.sum()
            cy = (_ys*image).sum()/image.sum()

            # calculate half light radius and avg lum at an annulus of 1 pixel width centered at the centroid of the image
            R, flag = _radius_at_fraction_of_total_circ(image, (cx, cy), image.shape[0]/2, 0.5)
            r = 0.5
            Rin, Rou = R-r, R+r
            if extraprints: print('R half light (px):', R)
            assert flag == 0
            an = photutils.aperture.CircularAnnulus((cx, cy), Rin, Rou)
            an_sum = np.abs(an.do_photometry(image, method='exact')[0][0])
            avg_lum = an_sum/(np.pi*(Rou*Rou - Rin*Rin))
            if extraprints: print("Average lum:", avg_lum) # at effective radius
        except:
            return Exception("error-code-7")

        # convert image to counts
        gain = segoptions["add_noise"]**2/avg_lum  # amplitude is 1/avg_lum
        if extraprints: print('Gain:', gain)
        
        counts = image*gain
        
        if segoptions['discretize']:
            exposure_time_sec = 1
            counts = np.around(counts * exposure_time_sec).astype(int)

        # add noise
        noise = np.random.poisson(lam=segoptions["add_noise"], size=counts.shape)
        counts += noise
        if extraprints: print('Signal to noise at the brightest pixel:', np.max(counts)/segoptions["add_noise"])
        if extraprints: print('Signal to noise at the least bright pixel:', np.min(counts)/segoptions["add_noise"])
        
        image = counts
        #OLD METHOD:: image += (np.max(image) / float(segoptions["add_noise"])) * np.random.standard_normal(size=(subim.npix, subim.npix))
    
    # --- subtract background ---
    if background_subtract:
        from photutils.background import Background2D, MedianBackground
        bkg_estimator = MedianBackground()
        bkg = Background2D(image, tuple(np.array(image.shape,dtype=int)//8), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        image -= bkg.background  # subtract the background
        if extraprints: print("Background RMS stats:", scipy.stats.describe(np.ravel(bkg.background_rms)))
    
    # display
    if not nodisplay:
        imagenorm = simple_norm(image, stretch='log', log_a=10000)
        if segoptions["add_noise"] is not None:
            axes[0, 1].imshow(image, origin='lower', cmap='gray', norm=imagenorm)
            axes[0, 1].set_title('{}Image + noise (flux)'.format('Background-subtracted ' if background_subtract else ''))
        else: 
            axes[0, 1].imshow(np.log10(image - np.min(image)), origin='lower', cmap='gray')
            axes[0, 1].set_title('{}Image (mag)'.format('Background-subtracted ' if background_subtract else ''))
    
    
    # --- segmentation ---
    threshold =  photutils_detect_threshold(image, segoptions["seg_thres"])
    segm = photutils_detect_sources(image, threshold, segoptions["npixels"])  
    if segm is None:
        return Exception("error-code-5")  # no sources detected
    
    if not nodisplay:
        axes[1, 0].imshow(segm.data, origin='lower')
        axes[1, 0].set_title('Segmentation Image')
#         if bhmergerpos_inpix is not None:
#             axes[1, 0].scatter(bhmergerpos_inpix[0], bhmergerpos_inpix[1], marker='x',color='red', label="BH(s)")
#             axes[1, 0].legend()
    
    # --- choosing the segmap ---
    def pick_mainlabel(segmap, bhmergerpos_inpix=None, verbose=False):
        """pick the main label for the given segmentation"""
        if bhmergerpos_inpix is not None:
            # choose based on the position the black holes . . .
            label = segmap.data[bhmergerpos_inpix[1], bhmergerpos_inpix[0]]
            if label != 0:
                if verbose: print("Chose the segmentation that contained the BHs.")
                return label
            
            # if bhs are not in a segmentation:
            print("WARNING: BHs are not in a segmentation, removing this from the sample.")
            return Exception("error-code-1")
        elif assert_check_black_holes and check_black_holes:
            print("WARNING: BH position information cannot be found.")
            return Exception("error-code-2")
        else:
            # otherwise, choose based on the size of the segmentation (future: and its proximity to the center? or maximum flux?)
            if verbose: print("Chose the largest segmentation.")
            return np.argmax(segmap.areas) + 1
        
    # choose label
    mainlabel = pick_mainlabel(segm, bhmergerpos_inpix, verbose=extraprints)
    if type(mainlabel) == Exception: # bh is outside of a segmentation
        return mainlabel
    
    # mask out all other labels from the image
    mask = np.isin(segm.data, [l for l in segm.labels if l != mainlabel])
    segm.keep_label(mainlabel)

    # --- deblending ---
    if deblend:
        deblend_kernel=astropy.convolution.Gaussian2DKernel(
            x_stddev=segoptions['dblnd_kernel_sigma'],  #max_width/subim.w
            y_stddev=segoptions['dblnd_kernel_sigma'], 
            x_size=segoptions['dblnd_kernel_size'], 
            y_size=segoptions['dblnd_kernel_size']
        )
        segm_deblend = photutils_deblend_sources(convolve(image, deblend_kernel), segm, npixels=segoptions["npixels"], 
                                                 nlevels=segoptions["dblnd_nlevels"], 
                                                 contrast=segoptions["dblnd_contrast"])
        if extraprints: print("Gaussian smoothing std.dev.:", max_width/subim.w)
        
        if not nodisplay:
            axes[2, 0].imshow(segm_deblend.data, origin='lower')
            axes[2, 0].set_title('Deblending Image')
    
        # pick the right deblend segmap
        mainlabel_deblend = pick_mainlabel(segm_deblend, bhmergerpos_inpix, verbose=False)
        if mainlabel_deblend is None:  # bh is outside of a segmentation
            return None
        
        # mask out all other labels from the image
        mask_deblend = np.isin(segm_deblend.data, [l for l in segm_deblend.labels if l != mainlabel_deblend])
        mask_deblend = np.logical_or(mask, mask_deblend)
        segm_deblend.keep_label(mainlabel_deblend)
    
    # --- regularization ---
    if segoptions['regularize'] is not None:
        segm = photutils.segmentation.SegmentationImage((ndi.uniform_filter(np.float64(segm.data), size=segoptions['regularize']) > 0.5).astype(int))
        segm_deblend = photutils.segmentation.SegmentationImage((ndi.uniform_filter(np.float64(segm_deblend.data), size=segoptions['regularize']) > 0.5).astype(int))
    
    # --- plot the chosen (possibly regularized) segmaps ---
    if not nodisplay:
        axes[1, 1].imshow(segm.data, origin='lower')
        if bhmergerpos_inpix is not None:
            axes[1, 1].scatter(bhmergerpos_inpix[0], bhmergerpos_inpix[1], marker='x',color='red', label="BH")
            axes[1, 1].legend()
        axes[1, 1].set_title('Chosen {}Segmap'.format("Regularized " if segoptions['regularize'] is not None else ''))
    
        if deblend:
            axes[2, 1].imshow(segm_deblend.data, origin='lower')
            if bhmergerpos_inpix is not None:
                axes[2, 1].scatter(bhmergerpos_inpix[0], bhmergerpos_inpix[1], marker='x',color='red', label="BH")
                axes[2, 1].legend()
            axes[2, 1].set_title('Chosen {}Deblend'.format("Regularized " if segoptions['regularize'] is not None else ''))
    
    # -- finish up plot ---
    if not nodisplay:
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        SpectralImage.finish_up_plot(output_filepath=outputfiles[0])
    
    # --- TEMP: testing ---
#     sc = photutils.segmentation.SourceCatalog(image, segm)
#     if extraprints:
#         print("Labels:", sc.labels)
#         print("Gini:", sc.gini)
#         print("Label in the center:", segm.data[tuple(np.array(segm.data.shape, dtype=int)//2)])
#         print("Gini of the center segmap:", sc.gini[np.where(sc.labels == segm.data[tuple(np.array(segm.data.shape, dtype=int)//2)])])
#         print("Local Background", sc.local_background)
#         print()
    
#     if deblend:
#         sc2 = photutils.segmentation.SourceCatalog(image, segm_deblend)
#         if extraprints:
#             print("Labels:", sc2.labels)
#             print("Gini:", sc2.gini)
#             print("Label in the center:", segm_deblend.data[tuple(np.array(segm_deblend.data.shape, dtype=int)//2)])
#             print("Gini of the center segmap:", sc2.gini[np.where(sc2.labels == segm_deblend.data[tuple(np.array(segm_deblend.data.shape, dtype=int)//2)])])
#             print("Local Background", sc2.local_background)
    
    # --- get morphs ---    
    source_morphs = []
    
    try:
        if not deblend or two_morphs:
            source_morphs.append(statmorph.source_morphology(image, segm, mask=mask, psf=psf, **morphoptions)[0])
            if not nodisplay:
                plt.clf()
                fig = make_figure(source_morphs[-1])
                fig.patch.set_facecolor('white')
                SpectralImage.finish_up_plot(output_filepath=outputfiles[1])

        if deblend:
            if extraprints: print('Using morph from deblending.')
            source_morphs.append(statmorph.source_morphology(image, segm_deblend, mask=mask_deblend, psf=psf, **morphoptions)[0])
            if not nodisplay:
                plt.clf()
                fig = make_figure(source_morphs[-1])
                fig.patch.set_facecolor('white')
                SpectralImage.finish_up_plot(output_filepath=outputfiles[2])
    except astropy.modeling.fitting.NonFiniteValueError:
        return Exception("error-code-3")
    except IndexError:
        return Exception("error-code-6")
    except:
        return Exception("error-code-99")
    
    plt.close('all')
    del segm
    del segm_deblend
    del subim
    gc.collect()
    print('Time: %g s.' % (time.time() - start))
    print()
    
    return source_morphs




# ---------------ADJUSTMENT TO SMOOTHNESS --------------------------
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
import scipy.ndimage as ndi
import warnings
import photutils

@lazyproperty
def _smoothness(self):
    """
    Calculate smoothness (a.k.a. clumpiness) as defined in eq. (11)
    from Lotz et al. (2004). Note that the original definition by
    Conselice (2003) includes an additional factor of 10.
    """
    image = self._cutout_stamp_maskzeroed

    # Exclude central region during smoothness calculation:
    r_in = self._petro_fraction_cas * self.rpetro_circ
    r_out = self._petro_extent_cas * self.rpetro_circ
    ap = photutils.aperture.CircularAnnulus(self._asymmetry_center, r_in, r_out)

    boxcar_size = max(int(self._petro_fraction_cas * self.rpetro_circ), 2)   # CHANGE IS HERE!!!! (added max with 2)
    image_smooth = ndi.uniform_filter(image, size=boxcar_size)

    image_diff = image - image_smooth
    image_diff[image_diff < 0] = 0.0  # set negative pixels to zero

    ap_flux = ap.do_photometry(image, method='exact')[0][0]
    ap_diff = ap.do_photometry(image_diff, method='exact')[0][0]

    if ap_flux <= 0:
        warnings.warn('[smoothness] Nonpositive total flux.',
                      AstropyUserWarning)
        self.flag = 2
        return -99.0  # invalid

    if self._sky_smoothness == -99.0:  # invalid skybox
        S = ap_diff / ap_flux
    else:
        S = (ap_diff - ap.area*self._sky_smoothness) / ap_flux

    if not np.isfinite(S):
        warnings.warn('Invalid smoothness.', AstropyUserWarning)
        self.flag = 2
        return -99.0  # invalid

    return S
    
statmorph.SourceMorphology.smoothness = _smoothness
# ------------------------------------------------------------------

# ### Run on all halos

# In[16]:


# Morphology measurements
def m_gini(morphs):
    return morphs[0].gini

def m_m20(morphs):
    return morphs[0].m20

def m_C(morphs):
    return morphs[0].concentration

def m_A(morphs):
    return morphs[0].asymmetry

def m_S(morphs):
    return morphs[0].smoothness

def m_As(morphs):
    return morphs[0].shape_asymmetry

def m_sersic_n(morphs):
    return morphs[0].sersic_n

def m_M(morphs):
    return morphs[0].multimode

def m_I(morphs):
    return morphs[0].intensity

def m_D(morphs):
    return morphs[0].deviation

def m_A0(morphs):
    return morphs[0].outer_asymmetry


# Contructed metrics
def m_merge_stat(morphs):
    return morphs[0].gini_m20_merger

def m_bulge_stat(morphs):
    return morphs[0].gini_m20_bulge

def m_sersic_logn(morphs):
    return np.log10(morphs[0].sersic_n)

# def m_Nevin_major_merger(morphs):  # it was done on standardized data, so this doesnt really work
#     return (0.69*m_gini(morphs) + 3.84*m_C(morphs) + 5.78*m_A(morphs) + 
#             13.14*m_As(morphs) - 3.68*m_gini(morphs)*m_As(morphs) - 
#             6.5*m_C(morphs)*m_As(morphs) - 6.12*m_A(morphs)*m_As(morphs) 
#             - 0.81) - 1.16

# def m_Nevin_minor_merger(morphs):
#     return (8.64*m_gini(morphs) + 14.22*m_C(morphs) + 5.21*m_A(morphs) + 
#             2.53*m_As(morphs) - 20.33*m_gini(morphs)*m_C(morphs) - 
#             4.32*m_A(morphs)*m_As(morphs) 
#             - 0.87) - 0.42

# Other metrics

def m_rpetro_circ(morphs):
    return morphs[0].rpetro_circ

def m_sersic_xc(morphs):
    return morphs[0].sersic_xc

def m_sersic_yc(morphs):
    return morphs[0].sersic_yc

def m_sersic_amplitude(morphs):
    return morphs[0].sersic_amplitude

def m_sersic_ellip(morphs):
    return morphs[0].sersic_ellip

def m_sersic_rhalf(morphs):
    return morphs[0].sersic_rhalf

def m_sersic_theta(morphs):
    return morphs[0].sersic_theta

def m_sky_mean(morphs):
    return morphs[0].sky_mean

def m_sky_sigma(morphs):
    return morphs[0].sky_sigma

def m_flag(morphs):
    return morphs[0].flag


# extract properties from morphs (put all desired properties in M_MEAS_FUNCTIONS and M_CONS_FUNCTIONS below)

M_MEAS_FUNCTIONS = {
    'Gini': m_gini,
    'M20': m_m20,
    'C': m_C,
    'A': m_A,
    'S': m_S,
    'As': m_As,
    'n': m_sersic_n,
    'M': m_M,
    'I': m_I,
    'D': m_D,
    'A0': m_A0,
}

M_CONS_FUNCTIONS = {
    'log n': m_sersic_logn,
    'merger stat': m_merge_stat,
    'bulge stat': m_bulge_stat,
#     'Nevin major': m_Nevin_major_merger,
#     'Nevin minor': m_Nevin_minor_merger,
}

M_OTHER_FUNCTIONS = {
    'rpetro': m_rpetro_circ,
    'sersic xc': m_sersic_xc,
    'sersic yc': m_sersic_yc,
    'sersic amplitude': m_sersic_amplitude,
    'sersic ellip': m_sersic_ellip,
    'sersic rhalf': m_sersic_rhalf,
    'sersic theta': m_sersic_theta,
    'sky mean': m_sky_mean,    # measure of noise: should be ~= S/N used (typically ~ 100 if not background subtracted)
    'sky sigma': m_sky_sigma,
    'flag': m_flag,
}

ALL_M_FUNCTIONS = M_MEAS_FUNCTIONS.copy()
ALL_M_FUNCTIONS.update(M_CONS_FUNCTIONS)
ALL_M_FUNCTIONS.update(M_OTHER_FUNCTIONS)

def extract_props_from_morphs(morphs):
    return [m_func(morphs) for m_func in ALL_M_FUNCTIONS.values()]


# In[17]:


# get halos to run on

sample_halos_events = set(list(zip(events_df['available step'], events_df['host halonumber'])))
major_mergers_events = set(list(zip(events_df[events_df['mass ratio'] >= 0.25]['host halonumber'].astype(int), events_df[events_df['mass ratio'] >= 0.25]['available step'].astype(int))))
minor_mergers_events = set(list(zip(events_df[events_df['mass ratio'] < 0.25]['host halonumber'].astype(int), events_df[events_df['mass ratio'] < 0.25]['available step'].astype(int))))

major_mergers = list(major_mergers_events)
minor_mergers = [mm for mm in minor_mergers_events if mm not in major_mergers_events]  # call a halo major if it has had a major event within 1 Gyr ("overwriting" any minors)

sample_halo_strs = ["{}_{}_{}".format(int(step),int(hn), int(i)) for hn, step in sample_halos for i in inclinations]
control_halo_strs = ["{}_{}_{}".format(int(step),int(hn), int(i)) for hn, step in control_halos for i in inclinations]
major_halo_strs = ["{}_{}_{}".format(int(step),int(hn), int(i)) for hn, step in major_mergers for i in inclinations]
minor_halo_strs = ["{}_{}_{}".format(int(step),int(hn), int(i)) for hn, step in minor_mergers for i in inclinations]

assert len(sample_halos_events) == len(major_mergers)+ len(minor_mergers)  # sanity check


available_sample_halo_strs_conv = [halo_str for halo_str in sample_halo_strs if os.path.exists("Image_Data/convolved.{}.hdf5".format(halo_str))]
available_control_halo_strs_conv = [halo_str for halo_str in control_halo_strs if os.path.exists("Image_Data/convolved.{}.hdf5".format(halo_str))]
available_major_halo_strs_conv = [halo_str for halo_str in available_sample_halo_strs_conv if halo_str in major_halo_strs]
available_minor_halo_strs_conv = [halo_str for halo_str in available_sample_halo_strs_conv if halo_str in minor_halo_strs]
assert len(available_major_halo_strs_conv) + len(available_minor_halo_strs_conv) == len(available_sample_halo_strs_conv) # otherwise, missing major/minor halo_str data

# morphs format: morphs[iden_tup] = [seg_morph, deb_morph]
# where iden_tup = (halo_str, filter_str) = (timestep_halonum_inclination_temp, _filter, ...)   (... = room to potentially expand later) 
# note: filter is assumed to be the *second* element (index 1) whenever iden_tup is used
available_sample_iden_strs_conv = [(str(halo_str), str(filter_)) for halo_str in available_sample_halo_strs_conv for filter_ in filters_conv]
available_control_iden_strs_conv = [(str(halo_str), str(filter_)) for halo_str in available_control_halo_strs_conv for filter_ in filters_conv]
available_major_iden_strs_conv = [(str(halo_str), str(filter_)) for halo_str in available_major_halo_strs_conv for filter_ in filters_conv]
available_minor_iden_strs_conv = [(str(halo_str), str(filter_)) for halo_str in available_minor_halo_strs_conv for filter_ in filters_conv]
assert len(available_major_iden_strs_conv) + len(available_minor_iden_strs_conv) == len(available_sample_iden_strs_conv)  # otherwise, missing major/minor halo_str data

assert len(filters_conv)*len(available_sample_halo_strs_conv) == len(available_sample_iden_strs_conv) # all halos are assumed to have all filters in filters_conv
assert len(filters_conv)*len(available_control_halo_strs_conv) == len(available_control_iden_strs_conv)
assert len(filters_conv)*len(available_major_halo_strs_conv) == len(available_major_iden_strs_conv)
assert len(filters_conv)*len(available_minor_halo_strs_conv) == len(available_minor_iden_strs_conv)

print(" -- POWDERDAY W/ DUST -- ")
print("(nsample, ncontrol, nmajor, nminor)")
print()
print("Distinct amount of halo images:")
print((len(available_sample_halo_strs_conv), len(available_control_halo_strs_conv), len(available_major_halo_strs_conv), len(available_minor_halo_strs_conv)))
print()
print("Distinct amount of monochromatic images (including filter data):")
print((len(available_sample_iden_strs_conv), len(available_control_iden_strs_conv), len(available_major_iden_strs_conv), len(available_minor_iden_strs_conv)))


# In[18]:


# Define what is good or bad data for each output of get_morphs
CLEANING_METHODS = [
    ("Morphs with their merging BHs outside of a segmentation", lambda morphs: type(morphs) == Exception and str(morphs) == "error-code-1"),
    ("Morphs without information on BH positions", lambda morphs: type(morphs) == Exception and str(morphs) == "error-code-2"),
    ("NonFiniteValueError (eg. 3905_18_1)", lambda morphs: type(morphs) == Exception and str(morphs) == "error-code-3"),
    ("OSError (eg. 4549_106_2)", lambda morphs: type(morphs) == Exception and str(morphs) == "error-code-4"),
    ("No segmentation sources found", lambda morphs: type(morphs) == Exception and str(morphs) == "error-code-5"),
    ("IndexError (eg. 6144_47_1)", lambda morphs: type(morphs) == Exception and str(morphs) == "error-code-6"),
    ("Error with Reff calculation", lambda morphs: type(morphs) == Exception and str(morphs) == "error-code-7"),
    ("Unknown Morph Error", lambda morphs: type(morphs) == Exception and str(morphs) == "error-code-99"),
    ("Morphs that are flagged flag > 1 AND flag_sersic == 1", lambda morphs: morphs[0].flag > 1 and morphs[0].flag_sersic == 1),
    ("Morphs that are flagged flag > 1 ONLY", lambda morphs: morphs[0].flag > 1 and morphs[0].flag_sersic != 1),
    ("Morphs that are flagged flag_sersic == 1 ONLY", lambda morphs: morphs[0].flag <= 1 and morphs[0].flag_sersic == 1),
]# second index was originally 1

def is_bad_morphs(morphs):
    for i, (cname, cfunc) in enumerate(CLEANING_METHODS):
        if cfunc(morphs):
            return i
    return 0


# In[19]:


DEFAULT_MORPH_PARAMS = dict(
    imagetype='convolved',
    check_black_holes=False, 
    assert_check_black_holes=False,
    nodisplay=False,   # True makes it about 2.5x faster, but there is no debug output..
    extraprints=False,
)
def make_morph_df(available_iden_strs, printtime=True, tracememory=False,
                  debug_output_path="Morph_Output_Images{}/".format(RUNIDEN),   # must include slash; set to None for 
                  morphparams=DEFAULT_MORPH_PARAMS):
    # log time
    if printtime: s_time = time.time()
    
    # for every available (halo_str, filter) pair (aka. iden_tup)
    morph_data = []
    morph_errors = [[] for _ in CLEANING_METHODS]
    for iden_tup in available_iden_strs:
        if tracememory: tracemalloc.start()  # memory profiling
        
        # compute the morph object
        morphs = get_morphs(iden_tup[0], iden_tup[1], 
                            outputfiles=(None if debug_output_path is None else 
                                        [debug_output_path+"segmentation_"+iden_tup[0]+'_'+iden_tup[1]+".png",  ## THIS IS UPDATED ONLY HERE (addition of +'_'+iden_tup[1] here and next 2 lines, aka filter added to the path)
                                         debug_output_path+"morph0_"+iden_tup[0]+'_'+iden_tup[1]+".png", 
                                         debug_output_path+"morph1_"+iden_tup[0]+'_'+iden_tup[1]+".png"]),
                            **morphparams)

        # check for any bad data (cleaning happens here)
        error_code = is_bad_morphs(morphs)
        if error_code != 0:  
            morph_errors[error_code].append(iden_tup)
            continue

        # add the data obtained to the pile
        morph_data.append([iden_tup[0], iden_tup[1], *extract_props_from_morphs(morphs)])
        
        # garbage collection
        del morphs
        gc.collect()
        
        if tracememory: 
            snapshot = tracemalloc.take_snapshot()  # memory profiling
            top_stats = snapshot.statistics('lineno')
            print("[ Top 10 ]")
            for stat in top_stats[:10]:
                print(stat)

    # contruct the dataframe
    morphs_df = pd.DataFrame(np.array(morph_data), columns=["halo_str","filter"]+list(ALL_M_FUNCTIONS.keys()))
    morphs_df = morphs_df.astype({k:'float' for k in ALL_M_FUNCTIONS.keys()})
    
    # log time
    if printtime:
        e_time = time.time()-s_time
        print("TOTAL TIME:", e_time, "secs =", e_time/60, "mins =", e_time/3600, "hours")
    
    #return df and the list of errors in the form of CLEANING_METHODS
    return morphs_df, morph_errors


# In[20]:


def delete_morph(self):
    try:
        del self._image
        del self._segmap
        del self.label
        del self._mask
        del self._weightmap
        del self._gain
        del self._psf
        del self._cutout_extent
        del self._min_cutout_size
        del self._n_sigma_outlier
        del self._annulus_width
        del self._eta
        del self._petro_fraction_gini
        del self._skybox_size
        del self._petro_extent_cas
        del self._petro_fraction_cas
        del self._boxcar_size_mid
        del self._niter_bh_mid
        del self._sigma_mid
        del self._petro_extent_flux
        del self._boxcar_size_shape_asym
        del self._sersic_maxiter
        del self._segmap_overlap_ratio
        del self._verbose
        del self.flag
        del self.flag_sersic
        del self._use_centroid
        del self._slice_stamp
        del self._mask_stamp_nan
        del self.num_badpixels
        del self.ymax_stamp
        del self.ymin_stamp
        del self.ny_stamp
        del self.xmax_stamp
        del self.xmin_stamp
        del self.nx_stamp
        del self._mask_stamp_badpixels
        del self._mask_stamp
        del self._mask_stamp_no_bg
        del self._cutout_stamp_maskzeroed_no_bg
        del self._centroid
        del self.xc_centroid
        del self._xc_stamp
        del self.yc_centroid
        del self._yc_stamp
        del self._cutout_stamp_maskzeroed
        del self._x_maxval_stamp
        del self._y_maxval_stamp
        del self._covariance_centroid
        del self._eigvals_centroid
        del self.ellipticity_centroid
        del self.elongation_centroid
        del self.orientation_centroid
        del self._diagonal_distance
        del self._rpetro_circ_centroid
        del self._slice_skybox
        del self._sky_asymmetry
        del self._asymmetry_center
        del self.xc_asymmetry
        del self.yc_asymmetry
        del self._covariance_asymmetry
        del self._eigvals_asymmetry
        del self.ellipticity_asymmetry
        del self.elongation_asymmetry
        del self.orientation_asymmetry
        del self.rpetro_circ
        del self.flux_circ
        del self.rpetro_ellip
        del self.flux_ellip
        del self._segmap_shape_asym
        del self.rmax_circ
        del self.rmax_ellip
        del self.rhalf_circ
        del self.rhalf_ellip
        del self.r20
        del self.r50
        del self.r80
        del self._segmap_gini
        del self.gini
        del self.m20
        del self.gini_m20_bulge
        del self.gini_m20_merger 
        del self.sky_sigma
        del self._weightmap_stamp
        del self.sn_per_pixel
        del self.concentration
        del self.asymmetry
        del self._sky_smoothness
        del self.smoothness
        del self._cutout_stamp_maskzeroed_no_bg_nonnegative
        del self._sorted_pixelvals_stamp_no_bg_nonnegative
        del self._segmap_mid 
        del self._cutout_mid
        del self._sorted_pixelvals_mid
        del self.multimode
        del self._cutout_mid_smooth
        del self._watershed_mid
        del self._intensity_sums
        del self.intensity
        del self.deviation
        del self.outer_asymmetry 
        del self.shape_asymmetry
        del self._sersic_model 
        del self.sersic_amplitude
        del self.sersic_rhalf 
        del self.sersic_n 
        del self.sersic_xc 
        del self.sersic_yc
        del self.sersic_ellip 
        del self.sersic_theta
        del self.sky_mean
        del self.sky_median
    except Exception as e:
        print("Caught Error: ", e)
    plt.close('all')
    gc.collect()

statmorph.SourceMorphology.__del__ = delete_morph


# #### Run it

# In[ ]:


_NODEBUG = True  # set to true to make it faster, but harder to debug

_morphparams = DEFAULT_MORPH_PARAMS.copy()
_morphparams["nodisplay"] = _NODEBUG

sample_morphs_df, sample_morphs_errors = make_morph_df(available_sample_iden_strs_conv, morphparams=_morphparams)
control_morphs_df, control_morphs_errors = make_morph_df(available_control_iden_strs_conv, morphparams=_morphparams)


# #### Clean up and save

# In[ ]:


#merge the two columns

# BAD METHOD:
#new_master_events = pd.concat([master_events, sample_morphs_df], axis=1, join="inner")
#new_master_control = pd.concat([master_control, control_morphs_df], axis=1, join="inner")

def merge_dfs_properly(df1, df2):
    ndf = pd.DataFrame(columns = [x for x in df1.columns if x not in df2.columns] + list(df2.columns))
    
    i = 0
    for row in df2.iterrows(): # for each halo-filter pair in df2
        _row = row[1]
        _data = df1[(df1["halo_str"] == _row["halo_str"]) & (df1["filter"] == _row["filter"])]
        
        for j in range(len(_data)): # for each event
            # make new row
            ndf.loc[i] = pd.Series(dict(_data.iloc[j]) | dict(_row))
            for c, dtype in set([(c2, df1[c2].dtype) for c2 in df1.columns]).union([(c2, df2[c2].dtype) for c2 in df2.columns]):
                ndf[c]=ndf[c].astype(dtype)
            i += 1
            
    return ndf

new_master_events = merge_dfs_properly(master_events, sample_morphs_df)
new_master_control = merge_dfs_properly(master_control, control_morphs_df)

#display
print("NEW EVENTS:")
display(new_master_events)
print()
print("NEW CONTROL:")
display(new_master_control)


save_custom_pd(new_master_events, "event_morphs{}.csv".format(RUNIDEN))
save_custom_pd(new_master_control, "control_morphs{}.csv".format(RUNIDEN))

# show errors stats
print('\n--- SAMPLE ERRORS ---\n')
tot_errs = sum([len(_err_halos) for _err_halos in sample_morphs_errors])
err_halo_strs, err_filters, err_reasons = [], [], []
for (_err_reason, _), _err_halos in zip(CLEANING_METHODS, sample_morphs_errors):
    print("    %s: %s %d cases (%.2f%% of sample)" % (_err_reason, ' '*(55 - len(_err_reason)), len(_err_halos), 100*len(_err_halos)/(len(sample_morphs_df)+tot_errs)))
    for _err_halo_str, _err_filter  in _err_halos:
        err_halo_strs.append(_err_halo_str)
        err_filters.append(_err_filter)
        err_reasons.append(_err_reason)

print('\nTOTAL: %d cases (%.2f%% of sample)\n' % (tot_errs, 100*tot_errs/(len(sample_morphs_df)+tot_errs)))
        
sample_errors_df = pd.DataFrame()
sample_errors_df['halo_str'] = err_halo_strs
sample_errors_df['filter'] = err_filters
sample_errors_df['error reason'] = err_reasons
sample_errors_df.to_csv('sample_errors'+RUNIDEN, index=False, header=True)

print('\n--- CONTROL ERRORS ---\n')
tot_errs = sum([len(_err_halos) for _err_halos in control_morphs_errors])
err_halo_strs, err_filters, err_reasons = [], [], []
for (_err_reason, _), _err_halos in zip(CLEANING_METHODS, control_morphs_errors):
    print("    %s: %s %d cases (%.2f%% of control)" % (_err_reason, ' '*(55 - len(_err_reason)), len(_err_halos), 100*len(_err_halos)/(len(control_morphs_df)+tot_errs)))
    for _err_halo_str, _err_filter  in _err_halos:
        err_halo_strs.append(_err_halo_str)
        err_filters.append(_err_filter)
        err_reasons.append(_err_reason)

print('\nTOTAL: %d cases (%.2f%% of control)\n' % (tot_errs, 100*tot_errs/(len(control_morphs_df)+tot_errs)))
        
control_errors_df = pd.DataFrame()
control_errors_df['halo_str'] = err_halo_strs
control_errors_df['filter'] = err_filters
control_errors_df['error reason'] = err_reasons
control_errors_df.to_csv('control_errors'+RUNIDEN, index=False, header=True)



