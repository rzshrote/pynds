import pytest
import numpy
from pynds.naive import krange, ndsort_naive
from matplotlib import pyplot
from pynds.relation import dominance_relationship

@pytest.fixture
def nindiv():
    yield 100

@pytest.fixture
def nobj():
    yield 2

@pytest.fixture
def xmat(nindiv, nobj):
    yield numpy.random.random((nindiv,nobj))

@pytest.fixture
def rankvec(nindiv):
    yield numpy.empty(nindiv, dtype = int)

def test_krange():
    exclude = 5
    for i in krange(0, exclude + 5, exclude):
        assert i != exclude

def test_ndsort_naive_inplace_True(xmat, rankvec):
    out = ndsort_naive(xmat, rankvec, inplace = True)
    assert id(out) == id(xmat)
    assert out.shape == xmat.shape

    # list the number of unique non-dominated frontiers
    nduniq = numpy.unique(rankvec)

    # for each front, assert that no other point dominates another point (all non-dominate each other)
    for front in nduniq:
        mask = (rankvec == front)
        ix = numpy.flatnonzero(mask)
        for i in ix:
            for j in ix:
                assert dominance_relationship(out[i,:], out[j,:]) == 0

    # for each successive front, assert that no member in the better front is dominated by any member in the worse front
    for fronti,frontj in zip(nduniq[:len(nduniq)-1], nduniq[1:]):
        maski = (rankvec == fronti)
        maskj = (rankvec == frontj)
        ixi = numpy.flatnonzero(maski)
        ixj = numpy.flatnonzero(maskj)
        for i in ixi:
            for j in ixj:
                # -1 == (x dom y); 0 == (x nondom y); 1 == (y dom x)
                # no point in the worse front should dominate any point in the better front
                assert dominance_relationship(out[i,:], out[j,:]) <= 0

    # make figure for visual inspection
    pyplot.scatter(out[:,0], out[:,1], c=rankvec, cmap='tab20')
    pyplot.savefig("test_naive_inplace_True.png")
    pyplot.close()

def test_ndsort_naive_inplace_False(xmat, rankvec):
    out = ndsort_naive(xmat, rankvec, inplace = False)
    assert id(out) != id(xmat)
    assert out.shape == xmat.shape

    # list the number of unique non-dominated frontiers
    nduniq = numpy.unique(rankvec)

    # for each front, assert that no other point dominates another point (all non-dominate each other)
    for front in nduniq:
        mask = (rankvec == front)
        ix = numpy.flatnonzero(mask)
        for i in ix:
            for j in ix:
                assert dominance_relationship(out[i,:], out[j,:]) == 0

    # for each successive front, assert that no member in the better front is dominated by any member in the worse front
    for fronti,frontj in zip(nduniq[:len(nduniq)-1], nduniq[1:]):
        maski = (rankvec == fronti)
        maskj = (rankvec == frontj)
        ixi = numpy.flatnonzero(maski)
        ixj = numpy.flatnonzero(maskj)
        for i in ixi:
            for j in ixj:
                # -1 == (x dom y); 0 == (x nondom y); 1 == (y dom x)
                # no point in the worse front should dominate any point in the better front
                assert dominance_relationship(out[i,:], out[j,:]) <= 0

    # make figure for visual inspection
    pyplot.scatter(out[:,0], out[:,1], c=rankvec, cmap='tab20')
    pyplot.savefig("test_naive_inplace_False.png")
    pyplot.close()
