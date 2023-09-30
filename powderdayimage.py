"""
Jaeden Bardati
"""


from typing import Type
import warnings
import colorsys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import scipy.optimize

from astropy.convolution import convolve
from astropy.visualization import make_lupton_rgb
from astropy.utils.exceptions import AstropyUserWarning

import photutils

import h5py

from PIL import Image, ImageEnhance


rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))

    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h += hue/360.
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))

    new_img = Image.fromarray(arr.astype('uint8'), 'RGBA')

    return new_img


def _fraction_of_total_function_circ(r, image, center, fraction, total_sum):
    """
    Helper function to calculate ``_radius_at_fraction_of_total_circ``.
    Taken from StatMorph (https://github.com/vrodgom/statmorph).
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
    Taken from StatMorph (https://github.com/vrodgom/statmorph).
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


def degrade_image(image, psf_fwhm=1., Reff_SN=100, gain=None, discretize=False, exposure_time=1.,
                  add_poisson_noise=True, poisson_lam=100, return_in_counts=False, extraprints=False):
    """
    Degrades an image by convolving with a PSF, estimating a gain, converting to counts, discretizing and adding Poisson noise.

    The parameters are as follows:
      - psf_fwhm: The sigma of the Gaussian PSF (None --> No PSF convolution).
      - gain: A value that when multiplied by the image, converts the image units to counts (None --> No manual gain, Reff_SN must be entered).
      - Reff_SN: The S/N value assumed at the Reff of the galaxy. This calculates gain for you (None --> Manual gain and noise_StN must be entered).
      - discretize: When set to True, it ensures that the image in counts is discrete (useful for low S/N images).
      - exposure_time: Roughly the exposure time of the image (i.e. higher value gives higher count value). Only relevant if discretize is True.
      - add_poisson_noise: When set to True, adds a background Poission noise appropriate to the S/N
      - poisson_lam: The lambda of the Poisson noise. Only relevant if gain is manually set (and thus Reff_SN is not estimated).
      - return_in_counts: When True, returns the image in counts, not original units.
    """
    if gain is None and Reff_SN is None: 
        raise ValueError('You cannot return in counts without assuming some gain. Set Reff_SN or gain manually.')
    if gain is not None and Reff_SN is not None:
        raise ValueError('You must either define the gain or approximate it by assuming a S/N at the effective radius.')

    if psf_fwhm is not None:  # convolve image with a PSF
        psf_size = np.shape(image)[0]//2-1  # on each side from the center, assuming square image
        y, x = np.mgrid[-psf_size:psf_size+1, -psf_size:psf_size+1]
        psf = np.exp(-(x**2 + y**2)/(2.0*(psf_fwhm/2.355)**2))  # note: psf_sigma = psf_fwhm/2.355
        psf /= np.sum(psf)
        image = convolve(image, psf)

    if gain is None:
        # calculate image centroid
        _xs = np.tile(np.arange(0, image.shape[0]),(image.shape[1],1))
        _ys = np.tile(np.arange(0, image.shape[1]).reshape((image.shape[1], 1)), image.shape[0])

        cx = (_xs*image).sum()/image.sum()
        cy = (_ys*image).sum()/image.sum()

        # calculate half light radius and avg lum at an annulus of 1 pixel width centered at the centroid of the image
        R, flag = _radius_at_fraction_of_total_circ(image, (cx, cy), image.shape[0]/2, 0.5)
        r = 0.5
        Rin, Rou = R-r, R+r
        if extraprints: print('R half light (px):', R)
        assert flag == 0, 'An error occurred while trying to find Reff.'
        an = photutils.aperture.CircularAnnulus((cx, cy), Rin, Rou)
        an_sum = np.abs(an.do_photometry(image, method='exact')[0][0])
        avg_lum = an_sum/(np.pi*(Rou*Rou - Rin*Rin))
        if extraprints: print("Average lum:", avg_lum) # at effective radius

        # estimate gain
        StN = Reff_SN
        gain = StN**2/avg_lum  # amplitude is 1/avg_lum
        if extraprints: print('Gain:', gain)
    else:
        StN = poisson_lam

    # convert image to counts
    counts = image*gain

    if discretize:  # force integer counts
        counts = np.around(counts * exposure_time).astype(int)

    if add_poisson_noise: # add noise
        noise = np.random.poisson(lam=StN, size=counts.shape)
        counts += noise
        if extraprints: print('Signal to noise at the brightest pixel:', np.max(counts)/StN)
        if extraprints: print('Signal to noise at the least bright pixel:', np.min(counts)/StN)

    if return_in_counts:  
        return counts   # return the image in counts
    
    image = counts/gain  # convert back to flux 
    return image



class MultiBandImage:
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
    def _zoom_image(array2d, zoom_factor, ret_real_zoom=False):
        if zoom_factor < 1:
            raise Exception("The zoom factor must be greater than or equal to 1.")
        xsize, ysize = array2d.shape[0], array2d.shape[1]  # size (in number of pixel space)
        cx, cy = int(xsize/2.), int(ysize/2.) # center pixel
        xnhs, ynhs = int(xsize/(zoom_factor*2.)), int(ysize/(zoom_factor*2.))  # new half size
        if not ret_real_zoom:
            return array2d[cx-xnhs:cx+xnhs, cy-ynhs:cy+ynhs]
        return array2d[cx-xnhs:cx+xnhs, cy-ynhs:cy+ynhs], xsize/array2d[cx-xnhs:cx+xnhs, cy-ynhs:cy+ynhs].shape[0]  #assumes square and even pixels
    
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
        """Returns a SingleBandImage object of a given filter."""
        return SingleBandImage(self._filepath, filt, rotate_cc90=self.rotate_cc90, zoom_factor=zoom_factor, mask=mask, flip_axis=flip_axis, log=log)
    
    def plot_rgb(self, rfilt, gfilt, bfilt, xlim=None, ylim=None, title=None, fontsize=None, labelsize=None,
                 pre_power=0.3, stretch=0.5, Q=4, sat=0.25, sha=1.5, con=1.2, bri=1.2, hs=0,
                 psf_fwhm=1., Reff_SN=100, gain=None, discretize=False, add_poisson_noise=True, poisson_lam=100, exposure_time=1,
                 log=False, zoom_factor=None, output_filepath=None, return_image=False):
        """
        Plots the RGB image using 3 given filters. 
        If output_filepath is not None, it saves the result at that location (.png).
        To avoid plotting and just return a matrix containing the image, set return_image to True.'
        To zoom in to the center of the image, set zoom_factor > 1.

        To adjust the color scheme, play around with the following parameters:
            - pre_power: Power before passing to Lupton et al. function, 
            - stretch: Lupton et al. parameter, 
            - Q: Lupton et al. parameter, 
            - sat: Saturation adjustment to the image, 
            - sha: Sharpness adjustment to the image, 
            - con: Contrast adjustment to the image, 
            - bri: Brightness adjustment to the iamge, 
            - hs: Hue shift of the image.

        You can degrade the image to look like mock telescope images with the following parameters:
            psf_fwhm, Reff_SN, gain, discretize, add_poisson_noise, poisson_lam, exposure_time
            See degrade_image function (above) for details.
        """
        
        if log: print("Generating image . . .")

        prefoo = lambda i: degrade_image(i, psf_fwhm=psf_fwhm, Reff_SN=Reff_SN, gain=gain, discretize=discretize, 
                                         add_poisson_noise=add_poisson_noise, poisson_lam=poisson_lam, exposure_time=exposure_time)
        R = prefoo(self._get_filter_data(rfilt))
        G = prefoo(self._get_filter_data(gfilt))
        B = prefoo(self._get_filter_data(bfilt))

        if zoom_factor is not None and zoom_factor != 1:
            R, real_zoomR = MultiBandImage._zoom_image(R, zoom_factor, ret_real_zoom=True)
            G, real_zoomG = MultiBandImage._zoom_image(G, zoom_factor, ret_real_zoom=True)
            B, real_zoomB = MultiBandImage._zoom_image(B, zoom_factor, ret_real_zoom=True)
            assert real_zoomR == real_zoomG == real_zoomB, 'There is something wrong with the zooming code.'
            zoom_factor = real_zoomR

        norm_factor=1/np.max([R, G, B])
        foo = lambda x: ((x - np.min(x))*norm_factor)**pre_power
        rgb_image = make_lupton_rgb(foo(R), foo(G), foo(B), Q=Q, stretch=stretch)
        
        img_obj = Image.fromarray(np.uint8((rgb_image-np.min(rgb_image))/np.max(rgb_image)*255))
        
        img_obj = ImageEnhance.Color(img_obj).enhance(sat)       # saturation
        img_obj = ImageEnhance.Sharpness(img_obj).enhance(sha)    # sharpness
        img_obj = ImageEnhance.Contrast(img_obj).enhance(con)     # contrast
        img_obj = ImageEnhance.Brightness(img_obj).enhance(bri)   # brightness 
        img_obj = colorize(img_obj, hs)                            # hue shift
        
        rgb_image = np.array(img_obj, dtype=int)

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

    
class SingleBandImage(MultiBandImage):
    """A class that inherits from the MultiBandImage class."""
    
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
                   psf_fwhm=None, Reff_SN=None, gain=1., discretize=False, add_poisson_noise=False, poisson_lam=100, exposure_time=1, 
                   vmin=None, finish_plot=True):
        """
        Plots the monochromatic image. 
        If output_filepath is not None, it saves the result at that location (.png).
        """
        if log: print("Plotting image . . .")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        showndata = degrade_image(self.data, psf_fwhm=psf_fwhm, Reff_SN=Reff_SN, gain=gain, discretize=discretize, 
                                  add_poisson_noise=add_poisson_noise, poisson_lam=poisson_lam, exposure_time=exposure_time) # in ergs/s
        
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

