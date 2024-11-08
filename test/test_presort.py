import pytest
import numpy
from pynds.presort import presort_matrix

################
### Fixtures ###
################

@pytest.fixture
def nsoln():
    yield 100

@pytest.fixture
def nobj():
    yield 5

@pytest.fixture
def xmat(nsoln, nobj):
    yield numpy.random.random((nsoln,nobj)).round(1)

@pytest.fixture
def xmat_sorted(xmat, nobj):
    keys = tuple(xmat[:,nobj-1-i] for i in range(nobj))
    ix = numpy.lexsort(keys)
    yield xmat[ix]

#############
### Tests ###
#############

def test_presort_matrix_inplace_True(xmat, xmat_sorted):
    out = presort_matrix(xmat, inplace = True)
    assert id(out) == id(xmat)
    assert numpy.all(out == xmat_sorted)
    assert numpy.all(xmat == xmat_sorted)

def test_presort_matrix_inplace_False(xmat, xmat_sorted):
    tmp = xmat.copy()
    out = presort_matrix(xmat, inplace = False)
    assert id(out) != id(xmat)
    assert numpy.all(out == xmat_sorted)
    assert numpy.all(xmat == tmp)
