import numpy as np

from logarray import log_array
from logarray.testing import assert_logarray_allclose
import pytest

test_arrays = [np.array([1,2,3]), np.array([-1,-2,-3]), np.array([1,2,-3])]
@pytest.mark.skip('fail')
@pytest.mark.parametrize("regular", test_arrays)
def test_sum(regular):
    mylogarray = log_array(regular)
    assert np.sum(mylogarray), regular.sum()
    logarray_sum = np.log(np.sum(mylogarray))
    regular_sum = np.log(np.sum(regular))
    assert_logarray_allclose(logarray_sum, regular_sum)

