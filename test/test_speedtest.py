import timeit
import pytest
import numpy

# helper functions

def generate_matrices(nobj, nindiv):
    x = numpy.random.random((nindiv,nobj))
    front = numpy.empty(nindiv, dtype = int)
    dom = numpy.empty((nindiv,nindiv), dtype = int)
    rem = numpy.empty(nindiv, dtype = int)
    mask = numpy.empty(nindiv, dtype = bool)
    return x, front, dom, rem, mask

@pytest.fixture
def objectives():
    yield [2,4,8]

@pytest.fixture
def individuals():
    yield [100,200,400,800]

def test_speedtest_ndsort_naive1(objectives, individuals):
    stats = numpy.empty((len(objectives),len(individuals)), dtype = float)

    for i,obj in enumerate(objectives):
        for j,indiv in enumerate(individuals):
            x, front, dom, rem, mask = generate_matrices(obj, indiv)
            stats[i,j] = timeit.timeit(
                'ndsort_naive1(x, front, rem, mask)', 
                setup = 'from pynds.naive import ndsort_naive1',
                number = 1, 
                globals = locals()
            )
    
    print(stats)

def test_speedtest_ndsort_naive2(objectives, individuals):
    stats = numpy.empty((len(objectives),len(individuals)), dtype = float)

    for i,obj in enumerate(objectives):
        for j,indiv in enumerate(individuals):
            x, front, dom, rem, mask = generate_matrices(obj, indiv)
            stats[i,j] = timeit.timeit(
                'ndsort_naive2(x, front, dom, rem, mask)', 
                setup = 'from pynds.naive import ndsort_naive2',
                number = 1, 
                globals = locals()
            )
    
    print(stats)
