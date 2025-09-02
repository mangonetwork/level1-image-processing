# Level 1 Data Processing
# Includes:
#   - Histogram Equalization
#   - Background Removal
#   - Atmospheric Correction


class ImageProcessor:

    def __init__(self, config):
        
        self.config = config


    def atmospheric_correction(self):
        """Calculate atmospheric correction arrays"""

        # Atmospheric corrections are taken from Kubota et al., 2001
        # Kubota, M., Fukunishi, H. & Okano, S. Characteristics of medium- and
        #   large-scale TIDs over Japan derived from OI 630-nm nightglow observation.
        #   Earth Planet Sp 53, 741–751 (2001). https://doi.org/10.1186/BF03352402

        # calculate zenith angle

        za = np.pi / 2 - self.elevation * np.pi / 180.0

        # Kubota et al., 2001; eqn. 6

        self.vanrhijn_factor = np.sqrt(1.0 - np.sin(za) ** 2 * (RE / self.REha) ** 2)


        # Kubota et al., 2001; eqn. 7,8

        a = 0.2
        F = 1.0 / (np.cos(za) + 0.15 * (93.885 - za * 180.0 / np.pi) ** (-1.253))
        self.extinction_factor = 10.0 ** (0.4 * a * F)



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

