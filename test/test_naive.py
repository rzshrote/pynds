import pytest
import numpy
from pynds.naive import krange
from pynds.naive import ndsort_naive
from pynds.naive import ndsort_naive1
from pynds.naive import ndsort_naive2
from matplotlib import pyplot
from pynds.relation import dominance_relationship

################################################################################
################################ Test Fixtures #################################
################################################################################

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

@pytest.fixture
def dommat(nindiv):
    yield numpy.empty((nindiv,nindiv), dtype = int)

@pytest.fixture
def remvec(nindiv):
    yield numpy.empty(nindiv, dtype = int)

@pytest.fixture
def maskvec(nindiv):
    yield numpy.empty(nindiv, dtype = bool)

################################################################################
################################## Unit Tests ##################################
################################################################################

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

def test_ndsort_naive1(xmat, rankvec, remvec, maskvec):
    ndsort_naive1(xmat, rankvec, remvec, maskvec)

    # list the number of unique non-dominated frontiers
    nduniq = numpy.unique(rankvec)

    # for each front, assert that no other point dominates another point (all non-dominate each other)
    for front in nduniq:
        mask = (rankvec == front)
        ix = numpy.flatnonzero(mask)
        for i in ix:
            for j in ix:
                assert dominance_relationship(xmat[i,:], xmat[j,:]) == 0

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
                assert dominance_relationship(xmat[i,:], xmat[j,:]) <= 0

    # make figure for visual inspection
    # print(numpy.concatenate([rankvec[:,None],xmat], axis = 1))
    pyplot.scatter(xmat[:,0], xmat[:,1], c=rankvec, cmap='tab20')
    pyplot.savefig("test_naive1.png")
    pyplot.close()

def test_ndsort_naive2(xmat, rankvec, dommat, remvec, maskvec):
    ndsort_naive2(xmat, rankvec, dommat, remvec, maskvec)

    # list the number of unique non-dominated frontiers
    nduniq = numpy.unique(rankvec)

    # for each front, assert that no other point dominates another point (all non-dominate each other)
    for front in nduniq:
        mask = (rankvec == front)
        ix = numpy.flatnonzero(mask)
        for i in ix:
            for j in ix:
                assert dominance_relationship(xmat[i,:], xmat[j,:]) == 0

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
                assert dominance_relationship(xmat[i,:], xmat[j,:]) <= 0

    # make figure for visual inspection
    # print(numpy.concatenate([rankvec[:,None],xmat], axis = 1))
    pyplot.scatter(xmat[:,0], xmat[:,1], c=rankvec, cmap='tab20')
    pyplot.savefig("test_naive2.png")
    pyplot.close()
