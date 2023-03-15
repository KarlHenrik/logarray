=====
Usage
=====
The main goal of logarray is to provide a convenient way to write functions in log space for numerical stability.

The basic context is that logarray objects should look and behave like numpy arrays, but perform all computations in logspace. In general for any functions we should have that 


>>> from logarray import log_array
>>> from numpy.testing import assert_allclose
>>> import numpy as np
>>> a = np.arange(1, 5)
>>> b = np.full(4, 3)
>>> result = (a+b)*2/a
>>> a, b = (log_array(a), log_array(b))
>>> log_result = (a+b)*2/a
>>> assert_allclose(np.log(log_result), np.log(result))
       

To use LogArray in a project::

    import logarray

