# Level 1 Data Processing
# Includes:
#   - Histogram Equalization
#   - Background Removal
#   - Atmospheric Correction

import argparse
import configparser
import logging
#import multiprocessing
import pathlib
import os
import sys
import time

import h5py
import numpy as np
from scipy.interpolate import griddata

#from . import imageops

import matplotlib.pyplot as plt

#if sys.version_info < (3, 9):
#    import importlib_resources as resources
#else:
#    from importlib import resources

RE = 6371.0  # Earth radius (m)

class ImageProcessor:

    def __init__(self, config):
        
        self.config = config



    def run(self, filename):
        """Run processing algorithm"""

        logging.debug(filename)

        # Read relevant fields from processed file
        image = h5py.File(filename)["ImageData"][:]
        mask = h5py.File(filename)["Mask"][:]

        # Read relevant configuration options
        remove_background = self.config.getboolean("PROCESSING", "REMOVE_BACKGROUND", fallback=True)
        atmoscorr = self.config.getboolean("PROCESSING", "ATMOSPHERIC_CORRECTION", fallback=False)
        histequal = self.config.getboolean("PROCESSING", "EQUALIZATION", fallback=False)
        uint8_out = self.config.getboolean("PROCESSING", "UINT8_OUT", fallback=False)

        # Remove background
        # This MUST be first, otherwise background won't be scaled correctly
        if remove_background:
            background = h5py.File(filename)["Background"][:]
            image = self.remove_background(image, background)

        # Apply atmospheric corrections
        if atmoscorr:
            elevation = h5py.File(filename)["Elevation"][:]
            ha = h5py.File(filename)["ProcessingInfo/Altitude"][()]
            image = self.atmospheric_correction(image, elevation, ha)

        # Equalize image
        if histequal:
            image = self.histogram_equalize(image)

        # Convert image to uint8
        if uint8_out:
            image = self.convert_uint8(image)

        # Apply mask
        try:
            image[np.broadcast_to(mask, image.shape)] = np.nan
        except ValueError:
            # If image has been converted to uint8, can't use NaNs
            image[np.broadcast_to(mask, image.shape)] = 0


        #c = plt.imshow(image[0], vmin=0, vmax=20000)
        c = plt.imshow(image[0])
        plt.colorbar(c)
        plt.show()


    def remove_background(self, image, background):
        """Subtract background from image"""

        image = image - background[:,None,None]
        # Correct any negative points after subtraction
        image[image < 0] = 0
    
        return image


    def atmospheric_correction(self, image, elevation, ha):
        """Apply atmospheric corrections"""

        # Atmospheric corrections are taken from Kubota et al., 2001
        # Kubota, M., Fukunishi, H. & Okano, S. Characteristics of medium- and
        #   large-scale TIDs over Japan derived from OI 630-nm nightglow observation.
        #   Earth Planet Sp 53, 741–751 (2001). https://doi.org/10.1186/BF03352402

        # calculate zenith angle

        za = np.pi / 2 - elevation * np.pi / 180.0

        REha = RE + ha

        # Kubota et al., 2001; eqn. 6

        vanrhijn_factor = np.sqrt(1.0 - np.sin(za) ** 2 * (RE / REha) ** 2)


        # Kubota et al., 2001; eqn. 7,8

        a = 0.2
        F = 1.0 / (np.cos(za) + 0.15 * (93.885 - za * 180.0 / np.pi) ** (-1.253))
        extinction_factor = 10.0 ** (0.4 * a * F)

        return image * vanrhijn_factor * extinction_factor


    def histogram_equalize(self, image, num_bins=10000):
        """Histogram Equalization to adjust contrast [1%-99%]"""
        
        contrast = self.config.getfloat("PROCESSING", "CONTRAST", fallback=100)
    
        image_array_1d = image.flatten()
    
        image_histogram, bins = np.histogram(image_array_1d, num_bins)
        image_histogram = image_histogram[1:]
        bins = bins[1:]
        cdf = np.cumsum(image_histogram)
    
        # spliced to cut off non-image area
        # any way to determine this dynamically?  How periminant is it?
        cdf = cdf[:9996]
    
        max_cdf = max(cdf)
        max_index = np.argmin(abs(cdf - contrast / 100 * max_cdf))
        min_index = np.argmin(abs(cdf - (100 - contrast) / 100 * max_cdf))
        vmax = float(bins[max_index])
        vmin = float(bins[min_index])
        low_value_indices = image_array_1d < vmin
        image_array_1d[low_value_indices] = vmin
        high_value_indices = image_array_1d > vmax
        image_array_1d[high_value_indices] = vmax
    
        return image_array_1d.reshape(image.shape)


    def convert_uint8(self, image):
        """Convert image array to uint8"""

        image = image * 255 / np.nanmax(image)
        return image.astype("uint8")


def process():
        elev_cutoff = self.config.getfloat("PROCESSING", "ELEVCUTOFF")
        remove_background = self.config.getboolean("PROCESSING", "REMOVE_BACKGROUND")
        contrast = self.config.getfloat("PROCESSING", "CONTRAST", fallback=100)
        histequal = self.config.getboolean("PROCESSING", "EQUALIZATION", fallback=False)
        vanrhijn = self.config.getboolean("PROCESSING", "VANRHIJN")
        extinction = self.config.getboolean("PROCESSING", "EXTINCTION")
        uint8_out = self.config.getboolean("PROCESSING", "UINT8_OUT", fallback=False)

        cooked_image = np.array(raw_image)

        # Does it matter which of these operations is performed first?
        if remove_background:
            cooked_image = imageops.background_removal(cooked_image)

        if histequal:
            cooked_image = imageops.equalize(cooked_image, contrast)

        # Apply atmopsheric correction
        if vanrhijn:
            new_image *= self.vanrhijn_factor

        if extinction:
            new_image *= self.extinction_factor


        # Renormalize each image and convert to int
        if uint8_out:
           new_image = (new_image * 255 / np.nanmax(new_image)).astype("uint8")



def parse_args():
    """Command line options"""

    parser = argparse.ArgumentParser(description="Create a level 2 mango data product")

    parser.add_argument(
        "-c", "--config", metavar="FILE", help="Alternate configuration file"
    )
    #parser.add_argument(
    #    "-f",
    #    "--filelist",
    #    metavar="FILE",
    #    help="A file with a list of .hdf5 file names",
    #)
    parser.add_argument(
        "-o",
        "--output",
        default="mango-l1.hdf5",
        help="Output filename (default is mango.hdf5)",
    )
#    parser.add_argument(
#        "-n", "--numproc", type=int, default=1, help="Number of parallel processes"
#    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbose output (repeat for more detail)",
    )

    #parser.add_argument("inputfiles", nargs="*")
    parser.add_argument("inputfile", metavar="FILE", help="Input file")

    return parser.parse_args()


#def find_inputfiles(args):
#    """Find input filenames"""
#
#    if args.filelist:
#        with open(args.filelist, encoding="utf-8") as f:
#            filenames = [line.strip() for line in f]
#            # filter blank lines
#            filenames = [line for line in filenames if line]
#    else:
#        filenames = args.inputfiles
#
#    return [filename for filename in filenames if os.path.exists(filename)]


def find_config(filename):
    """Load configuration file from package data"""

    with h5py.File(filename) as h5:
        station = h5["ImageData"].attrs["station"]
        instrument = h5["ImageData"].attrs["instrument"]

    # Placeholder for default config file location
    #   This function can be rewritten later
    config_dir = os.environ['MANGONETWORK_CONFIGS']

    config_file = os.path.join(config_dir, f"{station}-{instrument}.ini")

    logging.debug("Using package configuration file: %s", config_file)

    #name = f"{station}-{instrument}.ini"

    #logging.debug("Using package configuration file: %s", name)

    #return resources.files("mangonetwork.raw.data").joinpath(name).read_text()
    return config_file



def main():
    """Main application"""

    args = parse_args()

    fmt = "[%(asctime)s] %(levelname)s %(message)s"

    if args.verbose > 0:
        logging.basicConfig(format=fmt, level=logging.DEBUG)
    else:
        logging.basicConfig(format=fmt, level=logging.INFO)

    #inputs = find_inputfiles(args)

#    if not inputs:
#        logging.error("No input files found")
#        sys.exit(1)

    #logging.debug("Processing %d files", len(inputs))
    #logging.debug("Number of processes: %d", args.numproc)

    if args.config:
        logging.debug("Alternate configuration file: %s", args.config)
        if not os.path.exists(args.config):
            logging.error("Config file not found")
            sys.exit(1)
        with open(args.config, encoding="utf-8") as f:
            contents = f.read()
    else:
        #contents = find_config(inputs[0])
        contents = find_config(args.inputfile)

    config = configparser.ConfigParser()
    #config.read_string(contents)
    config.read(contents)

#    # Make whether or not to multiprocess an option
#    with multiprocessing.Pool(
#        processes=args.numproc, initializer=worker_init, initargs=(config,)
#    ) as pool:
#        results = pool.map(worker, inputs, chunksize=1)

    processor = ImageProcessor(config)
    processor.run(args.inputfile)
    #for filename in inputs:
    #    processor.run(filename)

    #output_file = pathlib.Path(args.output)
    #write_to_hdf5(output_file, config, results)

    sys.exit(0)


if __name__ == "__main__":
    main()
