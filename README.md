# powderday-image
A repository of convenience python functions and classes that handle the image output of the Powderday radiative transfer simulation.

### Requirements
numpy, matplotlib, astropy, pandas, photutils, h5py, [Hyperion](http://www.hyperion-rt.org/)

### Description
The `powderdayimage.py` file contains the following convenience functions and classes:

- `MultiBandImage` : Class for loading and accessing an hdf5 Powderday file, including displaying the image.
- `SingleBandImage` : Class for loading and accessing a single band in an hdf5 Powderday file, including displaying the image.
- `PowderdayDataCube`: Class for loading and accessing a datacube from Powderday (note: Powderday does not support kinematics, use SKIRT instead).
- `degrade_image`: Function that degrades a radiative transfer image with a PSF and some artificial noise to match observations.

### Examples
See the accompanying Jupyter Notebook `Powderday-output-access.ipynb` for example usages.
