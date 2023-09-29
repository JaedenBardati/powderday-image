# powderday-image
A repository of convenience python functions and classes that handle the image output of the Powderday radiative transfer simulation.

### Requirements
numpy, matplotlib, astropy, pandas, photutils, h5py, [Pillow](https://pypi.org/project/Pillow/), [Hyperion](http://www.hyperion-rt.org/)

### Description
The `powderdayimage.py` file contains the following convenience functions and classes:

- `MultiBandImage` : Class for loading and accessing an hdf5 Powderday file, including displaying the image.
- `SingleBandImage` : Class for loading and accessing a single band in an hdf5 Powderday file, including displaying the image.
- `degrade_image`: Function that degrades a radiative transfer image with a PSF and some artificial noise to match observations.

The `powderdaydatacube.py` file contains the following:
- `PowderdayDataCube`: Class for loading and accessing a datacube from Powderday (note: Powderday does not support kinematics, use SKIRT instead).

There is also `generate_filter_map.py` which is example code that generates a filter map between the filter output of Powderday and the correct filters. This is necessary to correct for an error in Powderday where the filter images are seemingly scrambled.

### Examples
See the accompanying Jupyter Notebook `Powderday-output-access.ipynb` for example usages.
