# *flashcurve*
## A machine-learning approach for the fast generation of adaptive-binning for lightcurves with Fermi-LAT gamma-ray data

### Notes
*flashcurve* is a Python tool which can be set up on any machine with Python version >= 3.7

It should mainly be used to create adaptive time bins with constant significance (TS) of Fermi-LAT gamma-ray data, but can also be used to make images of data within time bins (see the *flashcurve* [paper](https://www.sciencedirect.com/science/article/pii/S2213133725000101?via%3Dihub)). One can also use *flashcurve* to predict the TS of these individual images.

*flashcurve* requires the TensorFlow Lite runtime to be installed separately. This can be done via
```
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

If this doesn't work, check the TensorFlow Lite [official installation guide](https://www.tensorflow.org/lite/guide/python). If you already have TensorFlow installed, then you can simply comment out `import tflite_runtime.interpreter as tflite` in `image_tools.py` and uncomment `import tensorflow.lite as tflite` before installation. 

Installation of *flashcurve* (after cloning this GitHub repo) is done via
```
pip3 install <path to flashcurve repo folder>
```

Additionally, this repo contains a `fermi_tools.py` script with methods which require the Fermipy package. This script is optional for creating basic lightcurves with the generated time bins. For information on installing and using Fermipy, go to the [Fermipy wiki](https://fermipy.readthedocs.io/en/latest/).

### Usage

Check the `lc_search_example.py` script for a detailed example of usage. 

Check the `make_image_example.ipynb` Jupyter notebook for an example of generating, printing and making a TS prediction for a standalone image of photon events within a single time bin. 

### Acknowledgements

_Theo Glauch_ (project supervisor)

_Narek Sahakyan_ & _Paolo Giommi_ (lightcurve consultation)



[pyLCR](https://github.com/dankocevski/pyLCR)

---

PLEASE do not forget to cite the original paper: [Link](https://www.sciencedirect.com/science/article/pii/S2213133725000101?via%3Dihub)

Enjoy creating adaptive time bins :)

