"""
Jaeden Bardati
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import astropy.units as u

from hyperion.model import ModelOutput


class PowderdayDataCube:
    """
    A wrapper class of convience methods for accessing the data cubes
    acquired from Hyperion. It includes a way to extract pixel spectra at each pixel.

    This class loads the "rtout.image" output files from Powderday.

    For more documentation on the ModelOutput class:
    https://buildmedia.readthedocs.org/media/pdf/hyperion2/latest/hyperion2.pdf
    
    Note that this assumes the images are square.
    """
    def __init__(self, filepath, lum_unit='erg/s', wav_unit='micron', dist_unit='kpc'):
        self.filepath, self.lum_unit, self.wav_unit, self.dist_unit = filepath, u.Unit(lum_unit), u.Unit(wav_unit), u.Unit(dist_unit)
        
        self._m = ModelOutput(filepath)
        self._image = self._m.get_image(units='ergs/s')
        self._img_unit_factor =  (u.erg/u.second).to(lum_unit)
        self._sorted_wav_idxs = np.argsort(self._image.wav) 
        self._npix, self._w = self._image.val.shape[1], self._image.x_max * (u.cm).to(dist_unit)

        self.wav = self._image.wav * (u.micron).to(wav_unit)                    # this array is not sorted by default (use self._sorted_wav_idxs to do so)
        self.pos = (np.arange(self._npix)+0.5)*self._w/self._npix - self._w/2   # if there's an error with positioning, it's probably here
        self.shape = (len(self.pos), len(self.pos), len(self.wav))
        assert self._image.val.shape[1:] == self.shape   # sanity check

    @property
    def sorted_wav(self):
        return self.wav[self._sorted_wav_idxs]

    def __getitem__(self, index):
        return self._image.val[0][index] * self._img_unit_factor

    def __setitem__(self, index, val):
        self._image.val[0][index] = val / self._img_unit_factor

    def wav_index(self, wav):
        """Returns the wavelength index nearest to the inputted wavelength."""
        return np.argmin(np.abs(wav - self.wav))

    def pos_index(self, pos):
        """Returns the position index nearest to the inputted position."""
        return np.argmin(np.abs(pos - self.pos))

    def get_image(self, wav):
        """Returns the 2d image at a given wavelength."""
        return self[:, :, self.wav_index(wav)]

    def get_convolved_image(self, filt):
        """Returns the 2d image for a given filter (2d array of wavelength-throughput pairs)."""
        image = np.zeros((self._npix, self._npix))
        for wav, thru in filt:
            image += self.get_image(wav) * thru
        return image

    def get_spectrum(self, x, y):
        """Returns an array containing the sorted wavelengths and 
        an array containing the associated spectrum values, given 
        the pixel-space position (x, y), with (0, 0) centered in 
        the lower left corner. """
        return self.sorted_wav, self[x, y, :][self._sorted_wav_idxs]

    def plot_image(self, wav, filename=None, title=None, cmap=plt.cm.viridis):
        """Plots the image at the nearest wavelength to wav."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        cax = ax.imshow(np.log(self.get_image(wav)), cmap=cmap, origin='lower', 
                        extent=[-self._w, self._w, -self._w, self._w])
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel('x ({})'.format(self.dist_unit._repr_latex_()))
        ax.set_xlabel('y ({})'.format(self.dist_unit._repr_latex_()))
        ax.set_title("Image with $\lambda$ = {} {}".format(wav, self.wav_unit._repr_latex_()) if title is None else title)
        plt.colorbar(cax, label='log Luminosity ({})'.format(self.lum_unit._repr_latex_()))

        if filename is not None: fig.savefig(filename, bbox_inches='tight', dpi=150)

    def plot_spectrum(self, x, y, filename=None, title=None):
        """Plots the spectrum for a given point."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        ax.plot(*self.get_spectrum(x, y))
        ax.set_xlabel("$\lambda$ ({})".format(self.wav_unit._repr_latex_()))
        ax.set_ylabel("Luminosity")
        ax.set_title("Spectrum at pixel ({},{})".format(x, y) if title is None else title)

        if filename is not None: fig.savefig(filename, bbox_inches='tight', dpi=150)


if __name__ == "__main__":
    """
    Example code:
    """
    
    # Load in the wrapper class
    dc = PowderdayDataCube("/scratch/jbardati/projectdata/powderday_output/5271_4/pd_converted_romulus_output.5271_4_1.rtout.image",
                     lum_unit='erg/s', wav_unit='micron', dist_unit='kpc')

    """
    print(dc[45, 50, 2])            # returns the flux at pixel (45, 50) from the lower left corner for the wavelength at index 2
    print(dc.shape)                 # returns the array shape of the data cube
    print()

    print(dc.wav[2])                # returns the wavelength (microns) at index 2
    print(dc.pos[45], dc.pos[50])   # returns the postion (kpc) at indices 45 and 50
    print(dc.wav_index(0.75))       # returns the dc.wav index at which the wavelength is closest to 0.75 (microns)
    print(dc.pos_index(4.5))        # returns the dc.pos pixel index at which the position is closest to 4.5 (kpc) from the center of the image
    # Note that the index functions are O(n) for the number of wavelengths and the number of positions respectively, so when dealing
    # with large data cubes, it is best to iterate over indices and use dc.wav and dc.pos to convert to real values (which is O(1))
    # Note that the wavelengths are not sorted by default, but the positions are. You can use dc.sorted_wav, but note dc[:, :, dc.wav_index(dc.sorted_wav[0])] != dc[:, :, 0].
    print()

    print(dc[dc.pos_index(-4.5), dc.pos_index(4.5), dc.wav_index(0.75)])  # returns the flux for 0.75 microns at (-4.5kpc, 4.5kpc) from the center

    wavelengths, fluxes = dc.get_spectrum(x=45, y=50)           # returns the sorted wavelength array (microns) and respective fluxes (ergs/s) at a given pixel
    dc.plot_spectrum(x=45, y=50, filename='pd_spectrum.png')    # plots the spectrum at pixel (50, 50) to a given filename

    image = dc.get_image(wav=0.75)                      # returns the 2d image containing the fluxes (ergs/s) for a given single wavelength (microns)
    dc.plot_image(wav=0.75, filename='pd_image.png')    # plots the image at a wavelength of 0.75 (microns) to a given filename
    """

    #### make make my own convolution with the datacube and a filter:
    import pandas as pd

    filters = ['roman_F146', 'castor_g', 'castor_uv']
    for i in range(len(filters)):

        filt = pd.read_csv('filters/%s_low_rez.filter' % filters[i], delim_whitespace=True, header=None).to_numpy()

        im = dc.get_convolved_image(filt)

        plt.figure()
        plt.imshow(im, cmap='gray', origin='lower')
        plt.savefig('datacube--manually_convolved_image--%s.png' %filters[i], bbox_inches='tight')

        print(filters[i])
        print(im)
        print()
