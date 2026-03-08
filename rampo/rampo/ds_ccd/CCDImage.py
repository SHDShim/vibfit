import os
import time
from PIL import Image
import fabio
import numpy.ma as ma
import numpy as np
# import matplotlib.pyplot as plt
import datetime
import collections
import re

from ..utils import make_filename, extract_extension
from ..model.SpeFile import SpeFile


class CCDImage(object):
    def __init__(self):
        self.img_filename = None
        self.img = None
        self.x_wavelength_raw = None
        # mask for self.img, not self.intensity_cake (cake image)
        self.mask = None 
        # self.intensity is for intersity of 1D pattern
        self.intensity = None
        self.tth = None
        # the following three are for cake
        self.intensity_cake = None
        self.tth_cake = None
        self.chi_cake = None

    def _set_direct_image_axes(self, x_axis=None):
        if self.img is None:
            self.x_wavelength_raw = None
            self.tth = None
            self.intensity = None
            self.tth_cake = None
            self.chi_cake = None
            self.intensity_cake = None
            return
        if x_axis is None:
            x_axis = np.arange(self.img.shape[1], dtype=float)
            self.x_wavelength_raw = None
        else:
            x_axis = np.asarray(x_axis, dtype=float)
            self.x_wavelength_raw = np.asarray(x_axis, dtype=float)
        y_axis = np.arange(self.img.shape[0], dtype=float)
        self.tth_cake = x_axis
        self.chi_cake = y_axis
        self.intensity_cake = np.asarray(self.img, dtype=float)
        self.tth = x_axis
        self.intensity = np.nansum(self.intensity_cake, axis=0)

    def load(self, img_filename):
        self.img_filename = img_filename
        ext = extract_extension(self.img_filename).lower()
        self.x_wavelength_raw = None
        if ext == 'spe':
            spe = SpeFile(self.img_filename)
            data = spe.img[0] if isinstance(spe.img, list) else spe.img
            self.img = np.asarray(data, dtype=float)[::-1]
            self._set_direct_image_axes(spe.x_calibration)
            print(str(datetime.datetime.now())[:-7],
                  ": Load CCD image from ", self.img_filename)
            return
        if ext == 'tif':
            data = Image.open(self.img_filename)
        elif ext == 'tiff':
            data = Image.open(self.img_filename)
        elif ext == 'h5':
            images = fabio.open(self.img_filename)
            data = images.data
        elif ext == 'mar3450':
            data_fabio = fabio.open(img_filename)
            data = data_fabio.data
        elif ext == 'cbf':
            data_fabio = fabio.open(img_filename)
            data = data_fabio.data
        self.img = np.array(data)[::-1]
        self._set_direct_image_axes()
        print(str(datetime.datetime.now())[:-7], 
            ": Load ", self.img_filename)

    def apply_excitation_wavelength(self, laser_wavelength_nm):
        if self.x_wavelength_raw is None:
            return False
        laser = float(laser_wavelength_nm)
        x_nm = np.asarray(self.x_wavelength_raw, dtype=float)
        x_shift = np.full_like(x_nm, np.nan, dtype=float)
        valid = np.isfinite(x_nm) & (x_nm > 0.0) & np.isfinite(laser) & (laser > 0.0)
        x_shift[valid] = 1.0e7 / laser - 1.0e7 / x_nm[valid]
        self.tth = np.asarray(x_shift, dtype=float)
        self.tth_cake = np.asarray(x_shift, dtype=float)
        return True

    def histogram(self):
        import matplotlib.pyplot as plt

        if self.img is None:
            return
        f, ax = plt.subplots(figsize=(10, 4))
        ax.hist(self.img.ravel(), bins=256, fc='k', ec='k')
        f.show()

    def show(self, clim=(0, 8e3)):
        import matplotlib.pyplot as plt
        
        f, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(self.img, origin="lower", cmap="gray_r", clim=clim)
        cbar = f.colorbar(cax, orientation='horizontal')
        f.show()

    def integrate_to_1d(self, azimuth_range=None, **kwargs):
        if self.img is None:
            raise ValueError("CCD image is not loaded.")
        y0 = 0
        y1 = self.img.shape[0]
        if azimuth_range is not None:
            y0 = int(np.floor(min(azimuth_range)))
            y1 = int(np.ceil(max(azimuth_range)))
            y0 = max(0, y0)
            y1 = min(self.img.shape[0], y1)
        if y1 <= y0:
            raise ValueError("Selected CCD row range is empty.")
        rows = self.img[y0:y1, :]
        if self.mask is not None:
            mask_rows = np.asarray(self.mask[y0:y1, :], dtype=bool)
            rows = ma.array(rows, mask=mask_rows)
            intensity = np.asarray(rows.mean(axis=0).filled(np.nan), dtype=float)
        else:
            intensity = np.asarray(np.nanmean(rows, axis=0), dtype=float)
        return np.asarray(self.tth, dtype=float), np.nan_to_num(intensity, nan=0.0)

    def integrate_to_cake(self, **kwargs):
        if self.img is None:
            raise ValueError("CCD image is not loaded.")
        t_start = time.time()
        self._set_direct_image_axes(self.tth)
        if self.mask is not None:
            self.intensity_cake = ma.array(self.intensity_cake, mask=self.mask).filled(0.0)
        print(str(datetime.datetime.now())[:-7],
            ": CCD refresh takes {0:.2f}s".format(time.time() - t_start))

    def get_pattern(self):
        if self.tth is None:
            return None, None
        else:
            return self.tth, self.intensity

    def get_cake(self):
        if self.tth_cake is None:
            return None, None, None
        else:
            return self.intensity_cake, self.tth_cake, self.chi_cake
        
    def get_img_zrange(self):
        if self.img is None:
            return None
        else:
            zmin = self.img.min()
            zmax = self.img.max()
            return [zmin, zmax]

    def set_mask(self, range):
        """
        Calculate mask array for self.img.
        Mask pixels below range[0] and pixels above range[1]
        """
        if (self.img is None):
            # here returns array used for mask without any masked points
            self.mask = None
            return
        if (range is None):
            self.mask = np.zeros_like(self.img, dtype=bool)
            return
        # print('set_mask', self.img.max(), range)
        masked = ma.masked_where(
            (self.img < range[0]) | (self.img > range[1]), self.img)
        self.mask = masked.mask
        #self.integrate_to_cake()

    def get_mask(self):
        """
        Get mask for self.img
        If there is no img or mask, then return None
        """
        if (self.img is None):
            self.mask = None
        return self.mask
        
    def get_mask_range(self):
        """
        Return the numeric range spanned by the *unmasked* pixels.

        Returns:
            [vmin, vmax] (floats) for the unmasked data, or None if
            self.img or self.mask is missing or there are no unmasked pixels.
        """
        if self.mask is None or self.img is None:
            return None

        # Ensure mask is a boolean array of same shape as img
        mask = self.mask
        try:
            # If mask came from a MaskedArray, it might be a boolean array or MaskedConstant
            if np.ma.isMaskedArray(mask):
                mask = mask.mask
        except Exception:
            pass

        mask = np.asarray(mask, dtype=bool)

        # Compute unmasked boolean selection
        unmasked_sel = ~mask

        # If no unmasked pixels, nothing to return
        if not np.any(unmasked_sel):
            return None

        # Extract unmasked data, ignore NaN/Inf safely
        unmasked_vals = self.img[unmasked_sel]
        # Mask invalid float values
        unmasked_vals = unmasked_vals[np.isfinite(unmasked_vals)]

        if unmasked_vals.size == 0:
            return None

        vmin = float(np.nanmin(unmasked_vals))
        vmax = float(np.nanmax(unmasked_vals))

        return [vmin, vmax]

    def write_to_npy(self, chi_filen_wo_ext_in_temp):
        """
        filen = base filename without extension
        """
        f_tth = chi_filen_wo_ext_in_temp + '.tth.cake.npy'
        f_chi = chi_filen_wo_ext_in_temp + '.chi.cake.npy'
        f_int = chi_filen_wo_ext_in_temp + '.int.cake.npy'

    def read_cake_from_tempfile(self, temp_dir=None):
        tth_filen, azi_filen, int_filen = \
            self.make_temp_filenames(temp_dir=temp_dir)
        if os.path.exists(tth_filen) and os.path.exists(azi_filen) and \
                os.path.exists(int_filen):
            self.tth_cake = np.load(tth_filen)
            self.chi_cake = np.load(azi_filen)
            self.intensity_cake = np.load(int_filen)
            return True
        else:
            return False

    def make_temp_filenames(self, temp_dir=None):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        tth_filen = make_filename(self.img_filename, 'tth.cake.npy',
                                  temp_dir=temp_dir)
        azi_filen = make_filename(self.img_filename, 'azi.cake.npy',
                                  temp_dir=temp_dir)
        int_filen = make_filename(self.img_filename, 'int.cake.npy',
                                  temp_dir=temp_dir)
        return tth_filen, azi_filen, int_filen

    def write_temp_cakefiles(self, temp_dir):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        tth_filen, azi_filen, int_filen = self.make_temp_filenames(
            temp_dir=temp_dir)
        np.save(tth_filen, self.tth_cake)
        np.save(azi_filen, self.chi_cake)
        np.save(int_filen, self.intensity_cake)


DiffImg = CCDImage
