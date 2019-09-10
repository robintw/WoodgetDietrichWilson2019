import rasterio
import numpy as np


class ImageToML:
    def __init__(self):
        pass

    def image_to_ml_input(self, filename, band_to_remove=None):
        self.input_filename = filename

        # Read image data and mask
        with rasterio.open(filename, 'r') as img:
            self.img_array = img.read()
            self.mask_array = img.dataset_mask()

        if band_to_remove is not None:
            self.img_array = np.delete(self.img_array, band_to_remove, 0)

        # Set all No Data values to NaN
        broadcasted_mask_arr = np.broadcast_to(self.mask_array == 0, self.img_array.shape)
        self.img_array[broadcasted_mask_arr] = np.nan
        self.img_array[self.img_array == -9999] = np.nan

        self.n_bands, self.n_x, self.n_y = self.img_array.shape

        # Reshape the image to n_bands cols, and x*y rows
        reshaped_image = self.img_array.reshape(self.n_bands, self.n_x * self.n_y).T

        # Get a bool for each row, stating whether there are any NaNs in the row
        nans = np.isnan(reshaped_image)
        any_nans_in_row = np.any(nans, axis=1)

        self.just_valid_data = reshaped_image[any_nans_in_row == False, :]
        self.indices_where_valid = np.where(any_nans_in_row == False)[0]

        return self.just_valid_data

    def ml_output_to_image(self, ml_output, output_filename):
        # Create a NaN array, same size as full reshaped image
        results = np.full(self.n_x * self.n_y, np.nan)

        # Fill in the ML outputs into that image
        results[self.indices_where_valid] = ml_output

        reshaped_results = results.reshape(self.n_x, self.n_y)

        with rasterio.open(self.input_filename, 'r') as input_img:
            profile = input_img.profile

        profile['count'] = 1
        profile['compress'] = 'DEFLATE'

        with rasterio.open(output_filename, 'w', **profile) as output_img:
            output_img.write(reshaped_results, 1)
