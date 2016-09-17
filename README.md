# BioStampRC™ Examples

This repository contains examples to help [BioStampRC™](https://mc10inc.com) users parse, plot, and analyze their data.

See the *examples/* directory for 
- Python Plotting Examples
- Matlab/Octave Plotting Examples

See the data processing directory for
- A data filtering class implemented in Python
- Code for calculating and applying affine transforms for accelerometer calibration

## Python Plotting Examples
To run Python examples, first install the respective requirements (e.g. `matplotlib` and `numpy`) by runnning:

```bash
pip install -r examples/python/plotting/requirements.txt
```

To run plotting code, just call:

```bash
# Update script with your data file's path first
python examples/python/plotting/plot_data_accel.py
```

## Matlab/Octave Plotting Examples

No additional dependencies are required. To run, just call:

```matlab
% Update script with your data file's path first
run examples/matlab/plotting/plot_data_accel.m
```

## Data Filtering Class

The file BioStampRCDataFilter.py in the data processing directory contains a Python class for applying various filters to 
BioStampRC data. The class provides a Butterworth FIR filter, an Elliptical IIR filter, a notch filter, and a median filter.
Other filter types can be added in a straightforward fashion. See the code for examples.

The class is designed to be imported into other code, but the script can also be run from the command line as follows:
```bash
python BioStampRCDataFilter.py inputFile.csv outputFile f_low f_high
```
where inputFile.csv is the file containing data to be filtered, outputFile is the header of the output (six example output
files are created), f_low is the low-frequency cutoff of the filter, and f_high is the high-frequency cutoff.

## Affine Transform

The script computeAffine.py derives an affine transform for the accelerometer based on calibration measurements. The script
applyAffine.py applies a previously-derived affine transform to measured data. Contact MC10 for more information about 
calibration.
