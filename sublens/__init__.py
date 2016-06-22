"""
General subhalo lensing framework.

Intended to be used to measure, predict and fit the <\Delta\Sigma> profiles
around subhalos (satelitte galaxies) embedded in local density peaks
 (a.k.a  "groups" and "clusters").


Contents:
----------

* I/O side:
    - wrapper for Erin's xshear program and output
    - viusalization shorthands (triangle plots, Kernel smoothed 2D contours...)

* Data side:
    - \Delta\Sigma stacking
    - Absolute magnitudes and k-correction
    - Spherical rotations

* Model side:
    - MCMC fit with object oriented likelihood definition
    - Deltasigma prediction with uniform front-end and switchable back-end:
        - pure python
        - python compiled with cython
        - pure C back end planned for future release
    - Multi-threaded model evaluation (in progress)

"""
