import joblib
import numpy as np
from .ImageToML import ImageToML

category_code_to_worst_case = {0: -2.0,
                               1: -1.0,
                               2: -0.5,
                               3: -0.2,
                               4: -0.1,
                               5: -0.05,
                               6: 0.05,
                               7: 0.1,
                               8: 0.2,
                               9: 0.5,
                               10: 1.0,
                               11: 6.5,
                               -1: np.nan}

category_code_to_best_case = {0: -1.0,
                              1: -0.5,
                              2: -0.2,
                              3: -0.1,
                              4: -0.05,
                              5: 0.0,
                              6: 0.0,
                              7: 0.05,
                              8: 0.1,
                              9: 0.2,
                              10: 0.5,
                              11: 1.0,
                              -1: np.nan}


def reclassify(arr, d):
    result = np.full_like(arr, 9999)
    for from_val, to_val in d.items():
        result[arr == from_val] = to_val
    
    return result


def run_raster_through_ml(raster_filename, classifier_filename,
                          best_case_output_filename,
                          worst_case_output_filename):
    I2ML = ImageToML()

    print('Loading raster data and masking')
    data_for_ml = I2ML.image_to_ml_input(raster_filename)

    classifier = joblib.load(classifier_filename)
    print('Running classification')
    results = classifier.predict(data_for_ml).astype(float)

    results_best_case = reclassify(results, category_code_to_best_case)
    results_worst_case = reclassify(results, category_code_to_worst_case)

    I2ML.ml_output_to_image(results_best_case, best_case_output_filename)
    I2ML.ml_output_to_image(results_worst_case, worst_case_output_filename)

    return (results_best_case, results_worst_case)