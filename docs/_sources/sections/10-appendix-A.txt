.. Appendix A
.. Citation: :cite:``
.. Footnote: [#]_ 

.. _appendix-a:

Appendix A: Installation
------------------------

To use all of the features described in this paper, at least version 0.9.0 of
Statsmodels must be used. Many of the features are also available in version
0.8.0. Some of the features not available in 0.8.0 include simulation
smoothing and the univariate filtering and smoothing method.

The most straightforward way to install the correct version of Statsmodels is
using pip. The following steps should be followed.

1. Install git. Instructions are available many places, for example at
   https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
2. Install the development version of Statsmodels using the following command:

::

    pip install git+git://github.com/statsmodels/statsmodels.git

At this point, the package should installed. If you have the Nose package
installed, you can test for a successful installation by running the following
command (this may take a few minutes):

::

    python -c "import statsmodels.tsa.statespace as ssm; ssm.test();"

There should be no failures (although a number of Warnings are to be expected).


Dependencies
^^^^^^^^^^^^

The Statsmodels library requires the "standard Python stack" of scientific
libraries:

- NumPy
- SciPy >= 0.17.1
- Pandas >= 0.18.1
- Cython >= 0.22.0
- Git (this is required to install the development version of Statsmodels)

There are also a few optional dependencies:

- Matplotlib; this is required for plotting functionality
- Nose; this is required for running the test suite
- IPython / Jupyter; this is required for running the examples or building the documentation
