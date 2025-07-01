# *flashcurve*
## A machine-learning approach for the fast generation of adaptive-binning for lightcurves with Fermi-LAT gamma-ray data

### News!

*flashcurve* has received a major update with a new time bin search algorithm using a bisection method.
This method offers greater efficiency, and eliminates the need for fine-tuning energy and source proximity filters. Check `flashcurve_bisection.pdf` for more details.

### Notes
*flashcurve* is a python tool which can be setup on any machine with python version >= 3.7

It should mostly be used to create adaptive time bins with constant significance (TS), but can also be used make images of data within time bins (see *flashcurve* paper). One can also predict the TS of these images.

*flashcurve* requires tensorflow-lite runtime to be installed seperately. This can be done via
```
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

If this doesn't work, check the tensorflow lite [official installation guide](https://www.tensorflow.org/lite/guide/python). If you already have tensorflow installed, then you can just simply comment out `import tflite_runtime.interpreter as tflite` in `image_tools.py` and uncomment `import tensorflow.lite as tflite` before installation. 

Installation of *flashcurve* (after cloning this github repo) is done via
```
pip install <path to flashcurve repo folder>
```

Additionally, this repo contains a `fermi_tools.py` script with methods which require the fermipy package. Using this script is optional for creating basic lightcurves with the generated time bins. For information on how to install and use fermipy go to the [Fermipy wiki](https://fermipy.readthedocs.io/en/latest/).

### Usage

Check the `lc_search_example.py` script for a detailed example of usage. 

Check `make_image_example.ipynb` jupyter notebook for an example of generating, printing and making a TS prediction for a standalone image of photon events within a single time bin. 



### Acknowledgements

_Theo Glauch_ (project supervisor)

_Narek Sahakyan_ & _Paolo Giommi_ (lightcurve consultation)



[pyLCR](https://github.com/dankocevski/pyLCR)

---

PLEASE do not forget to cite the original paper: https://doi.org/10.1016/j.ascom.2025.100937

Bibtex:
```
@article{GLAUCH2025100937,
title = {flashcurve: A machine-learning approach for the simple and fast generation of adaptive-binning light curves with Fermi-LAT data},
journal = {Astronomy and Computing},
volume = {51},
pages = {100937},
year = {2025},
issn = {2213-1337},
doi = {https://doi.org/10.1016/j.ascom.2025.100937},
url = {https://www.sciencedirect.com/science/article/pii/S2213133725000101},
author = {T. Glauch and K. Tchiorniy},
}
```

Enjoy creating adaptive time bins :)

