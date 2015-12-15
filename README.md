# Coherent-Point-Drift-Python
This is the course project of Medical Visualization. The main goal of this project is to reimplement Coherent Point Drift using python.This project is supervised by [Clinical Graphics](https://www.clinicalgraphics.com/)

## Schedule
**week1:** Working example (prototype)(**21/12/2015**)
**week2:** Refactored code (**28/12/2015**)
**week3:** Review (**4/01/2016**)
**week4:** Report (**11/01/2016**)

## Requriement

1. For evaluation, consider rigid and non-rigid separately, compare to ICP and Procrustes Analysis as baseline algorithms, and use the Intersection-Over-Union (IOU) as a scoring measure.

2. For production quality, write unit tests with py.test, check your code style with flake8, and garantuee maintainability by making your code easy to read and use. Document it according to numpydoc conventions. Make sure that your development environment is reproducible by using conda-env's environment.yml format.

## Reference
- https://flake8.readthedocs.org/en/latest/
- http://pytest.org/latest/
- https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
- https://github.com/conda/conda-env
- http://www.isi.uu.nl/Research/Publications/tmp/564.pdf (see equation 10 for the IOU metric)