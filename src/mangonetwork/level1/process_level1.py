# Level 1 Data Processing
import argparse
import configparser
import logging
import warnings
import pathlib
import os
import sys
import time
import datetime as dt

import h5py
import numpy as np
from skyfield.api import load, wgs84, utc
from skyfield.almanac import moon_phase


#import matplotlib.pyplot as plt
try:
    from mangonetwork.clouddetect.makePrediction import makePrediction  # Importing this module is slow.  Only do it if necessary?
except ImportError:
    warnings.warn("The cloud-detection package is not installed!  Cloud flags cannot be calculated.", ImportWarning)

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
        cloud_flag = self.config.getboolean("PROCESSING", "CLOUD_FLAG", fallback=False)
        uint8_out = self.config.getboolean("PROCESSING", "UINT8_OUT", fallback=False)

        # Remove background
        # This MUST be first, otherwise background won't be scaled correctly
        if remove_background:
            background = h5py.File(filename)["Background"][:]
            image = self.remove_background(image, background)

        # Apply atmospheric corrections
        if atmoscorr:
            elevation = h5py.File(filename)["Coordinates/Elevation"][:]
            ha = h5py.File(filename)["ProcessingInfo/Altitude"][()]
            image = self.atmospheric_correction(image, elevation, ha)

        # Equalize image
        if histequal:
            image = self.histogram_equalize(image)

        # Create cloud flag array
        if cloud_flag:
            utime = h5py.File(filename)["UnixTime"][:,0]
            station = h5py.File(filename)["ImageData"].attrs["station"]
            instrument = h5py.File(filename)["ImageData"].attrs["instrument"]
            raw_mango_dir = self.config.get("PROCESSING", "RAW_MANGO_DIR")
            self.cloudy = self.check_clouds(station, instrument, utime, raw_mango_dir)

        # Create moon array
        utime = h5py.File(filename)["UnixTime"][:,0]
        coord = h5py.File(filename)["SiteInfo/Coordinates"][:]
        self.moon_az, self.moon_el, self.moon_phase = self.get_moon(utime, coord[0], coord[1])

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
        #c = plt.imshow(image[0])
        #plt.colorbar(c)
        #plt.show()
        self.image = image
        self.input_file = filename


    def remove_background(self, image, background):
        """Subtract background from image"""

        # Subtract background from all time stamp
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


    def check_clouds(self, station, instrument, utime, raw_mango_dir):
        """Check if each image is cloudy or not"""

        cloudy = list()
        for ut in utime:
            time = dt.datetime.utcfromtimestamp(ut)
            raw_filename = f'{station}/{instrument}/raw/{time:%Y/%j/%H}/mango-{station}-{instrument}-{time:%Y%m%d-%H%M%S}.hdf5'
            raw_filepath = os.path.join(raw_mango_dir, raw_filename)
            res = makePrediction(instrument, filename=raw_filepath)
            cloudy.append(res)

        return cloudy


    def get_moon(self, utime, lat, lon):

        ts = load.timescale()
        
        # Set up skyfild times array
        times_dt = [dt.datetime.utcfromtimestamp(ut).replace(tzinfo=utc) for ut in utime]
        times = ts.from_datetimes(times_dt)

        # Load the JPL ephemeris DE421 (covers 1900-2050).
        eph = load('de421.bsp')

        # Calculate moon position
        earth, moon = eph['Earth'], eph['Moon']
        site = earth + wgs84.latlon(lat, lon)
        astrometric = site.at(times).observe(moon)
        el, az, d = astrometric.apparent().altaz()
        
        moon_az = az.degrees
        moon_el = az.degrees

        # Calculate moon phase
        phase = moon_phase(eph, times).degrees      # Does it make sense for this to be an array or one time for the whole night?

        return moon_az, moon_el, phase


    def write_to_hdf5(self, output_file):
    
        # Read relevant configuration options
        remove_background = self.config.getboolean("PROCESSING", "REMOVE_BACKGROUND", fallback=True)
        atmoscorr = self.config.getboolean("PROCESSING", "ATMOSPHERIC_CORRECTION", fallback=False)
        histequal = self.config.getboolean("PROCESSING", "EQUALIZATION", fallback=False)
        cloud_flag = self.config.getboolean("PROCESSING", "CLOUD_FLAG", fallback=False)
        uint8_out = self.config.getboolean("PROCESSING", "UINT8_OUT", fallback=False)

        output_file.parent.mkdir(parents=True, exist_ok=True)
    
        infile = h5py.File(self.input_file, "r")
    
        with h5py.File(output_file, "w") as f:
            infile.copy("SiteInfo", f)
            infile.copy("ProcessingInfo", f)
            infile.copy("Coordinates", f)
            infile.copy("DataQuality", f)
            infile.copy("UnixTime", f)
    
            images = f.create_dataset(
                "ImageData",
                data=self.image,
                compression="gzip",
                compression_opts=1,
            )
            images.attrs["Description"] = "pixel values for images"
            images.attrs["station"] = infile["ImageData"].attrs["station"]
            images.attrs["instrument"] = infile["ImageData"].attrs["instrument"]
            images.attrs["remove_background"] = remove_background
            images.attrs["atmospheric_correction"] = atmoscorr
            images.attrs["equalization"] = histequal
            images.attrs["uint8_out"] = uint8_out

            mp = f.create_dataset(
                    "DataQuality/MoonPhase", 
                    data=self.moon_phase, 
                    compression="gzip",
                    compression_opts=1
            )
            mp.attrs["Description"] = "phase of moon in degrees (0 = new moon; 180 = full moon)"

            ma = f.create_dataset(
                    "DataQuality/MoonAzimuth", 
                    data=self.moon_az, 
                    compression="gzip", 
                    compression_opts=1
            )
            ma.attrs["Description"] = "azimuth position of moon (degrees)"

            me = f.create_dataset(
                    "DataQuality/MoonElevation", 
                    data=self.moon_el, 
                    compression="gzip", 
                    compression_opts=1
            )
            me.attrs["Description"] = "elevation position of moon (degrees)"

            if cloud_flag:
                cf = f.create_dataset(
                    "DataQuality/CloudFlag", 
                    data=self.cloudy, 
                    compression="gzip", 
                    compression_opts=1
                )
                cf.attrs["Description"] = "sky is cloudy or clear (cloudy=True; clear=False)"
                cf.attrs["Size"] = "Nrecords"



#===================================================================================

def parse_args():
    """Command line options"""

    parser = argparse.ArgumentParser(description="Create a level 2 mango data product")

    parser.add_argument(
        "-c", "--config", metavar="FILE", help="Alternate configuration file"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mango-l1.hdf5",
        help="Output filename (default is mango.hdf5)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbose output (repeat for more detail)",
    )
    parser.add_argument("inputfile", metavar="FILE", help="Input file")

    return parser.parse_args()


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

    if args.config:
        logging.debug("Alternate configuration file: %s", args.config)
        if not os.path.exists(args.config):
            logging.error("Config file not found")
            sys.exit(1)
        with open(args.config, encoding="utf-8") as f:
            contents = f.read()
    else:
        contents = find_config(args.inputfile)

    config = configparser.ConfigParser()
    config.read(contents)

    processor = ImageProcessor(config)
    processor.run(args.inputfile)

    output_file = pathlib.Path(args.output)
    processor.write_to_hdf5(output_file)

    sys.exit(0)


if __name__ == "__main__":
    main()

