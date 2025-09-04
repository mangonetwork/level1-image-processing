# level1-image-processing
Process Level 1 mango data products

The resulting level 2 images may have the following modifications:

- Background Removal
- Atmospheric Correction (VanRhijn and Extinction Factor)
- Histogram Equalization
- Conversion to UINT8

Usage
=====

To run the level 1 processing:
```
mango-process-level1 <level-1-mango-file.hdf5>
```
To control exactly which modification options are applied, includ a config file with the `-c` flag.

Installation
============
This package can be cloned and installed with pip.
```
git clone https://github.com/mangonetwork/level1-image-processing.git
pip install level1-image-processing
```

