# powderday-image
A repository of convenience python functions and classes that handle the image output of the Powderday radiative transfer simulation.

### Requirements
numpy, matplotlib, h5py, astropy, pandas, [Hyperion](https://pypi.org/project/Hyperion/)

### Description
The `powderdayimage.py` file contains the following convenience functions and classes:

- `MultiBandImage` : Class for loading and accessing an hdf5 Powderday file, including displaying the image.
- `SingleBandImage` : Class for loading and accessing a single band in an hdf5 Powderday file, including displaying the image.

### Examples
See the accompanying Jupyter Notebook `Powderday-output-access.ipynb` for example usages.
