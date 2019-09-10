# Machine Learning code for Woodget, Dietrich and Wilson (2019)

This repository contains the code for the machine learning analyses carried out in the paper *Quantifying below-water fluvial geomorphic change: The implications of refraction correction, water surface elevations and spatially variable error* by Woodget, Dietrich and Wilson (2019).

I will update this README with a link to the paper once it has been published.

## Dependencies
The easiest way to install all the required dependencies is to use `conda` and run `conda env create -f environment.yml`.

If you can't do that for some reason, then the main dependencies are listed below:

 - python >=3.7
 - numpy
 - pandas
 - scikit-learn
 - gdal
 - rasterio
 - matplotlib
 - scipy
 - imbalanced-learn
 - seaborn
 - jupyter
 - tpot
 - tqdm

## Code explanation
The main library code is in the `ErrorML` folder. The main `ErrorML.py` code deals with loading and pre-processing the data, as well as creating various classifiers (including the final Gaussian Naive Bayes/PCA classifier used in the paper). The code in `ImageToML.py` deals with applying a scikit-learn classifier to an image using `GDAL` and `rasterio`.

The notebooks in the root of the repository each perform one of the analyses used in the work for the paper. The results from some of these notebooks weren't reported in the paper (eg. `Initial ML investigations.ipynb` and `Investigate PCA Parameters.ipynb`) and some are tidier than others. They all use the `ErrorML` library for data loading and preprocessing.

If you have any questions about this code then please raise an issue on this repository or email me at [robin@rtwilson.com](mailto:robin@rtwilson.com) and I will do my best to help.