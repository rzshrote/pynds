import pytest
import numpy
from pynds.relation import dominance_relationship
# from pynds.relation import maximizing_dominance_relationship

def test_minimizing_dominance_relationship_1d():
    # x dominates y
    x = numpy.array([0.0], dtype = float)
    y = numpy.array([1.0], dtype = float)
    assert dominance_relationship(x, y) == -1

    # x is nondominated by y
    x = numpy.array([0.0], dtype = float)
    y = numpy.array([0.0], dtype = float)
    assert dominance_relationship(x, y) == 0
    assert dominance_relationship(x, x) == 0
    assert dominance_relationship(y, y) == 0

    # x is dominated by y
    x = numpy.array([1.0], dtype = float)
    y = numpy.array([0.0], dtype = float)
    assert dominance_relationship(x, y) == 1

def test_minimizing_dominance_relationship_2d():
    # x dominates y
    x = numpy.array([0.0, 0.0], dtype = float)
    y = numpy.array([1.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == -1
    x = numpy.array([1.0, 0.0], dtype = float)
    y = numpy.array([1.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == -1

    # x is nondominated by y
    x = numpy.array([0.0, 0.0], dtype = float)
    y = numpy.array([0.0, 0.0], dtype = float)
    assert dominance_relationship(x, y) == 0
    x = numpy.array([1.0, 0.0], dtype = float)
    y = numpy.array([0.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == 0
    x = numpy.array([0.0, 1.0], dtype = float)
    y = numpy.array([1.0, 0.0], dtype = float)
    assert dominance_relationship(x, y) == 0
    assert dominance_relationship(x, x) == 0
    assert dominance_relationship(y, y) == 0

    # x is dominated by y
    x = numpy.array([1.0, 1.0], dtype = float)
    y = numpy.array([0.0, 0.0], dtype = float)
    assert dominance_relationship(x, y) == 1
    x = numpy.array([1.0, 1.0], dtype = float)
    y = numpy.array([0.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == 1

def test_minimizing_dominance_relationship_3d():
    # x dominates y
    x = numpy.array([0.0, 0.0, 0.0], dtype = float)
    y = numpy.array([1.0, 1.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == -1
    x = numpy.array([1.0, 0.0, 0.0], dtype = float)
    y = numpy.array([1.0, 1.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == -1
    x = numpy.array([0.0, 1.0, 0.0], dtype = float)
    y = numpy.array([1.0, 1.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == -1
    x = numpy.array([0.0, 0.0, 1.0], dtype = float)
    y = numpy.array([1.0, 1.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == -1

    # x is nondominated by y
    x = numpy.array([1.0, 0.0, 0.0], dtype = float)
    y = numpy.array([0.0, 1.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == 0
    x = numpy.array([0.0, 1.0, 0.0], dtype = float)
    y = numpy.array([1.0, 0.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == 0
    x = numpy.array([0.0, 0.0, 1.0], dtype = float)
    y = numpy.array([1.0, 1.0, 0.0], dtype = float)
    assert dominance_relationship(x, y) == 0
    assert dominance_relationship(x, x) == 0
    assert dominance_relationship(y, y) == 0

    # x is dominated by y
    x = numpy.array([1.0, 1.0, 1.0], dtype = float)
    y = numpy.array([0.0, 0.0, 0.0], dtype = float)
    assert dominance_relationship(x, y) == 1
    x = numpy.array([1.0, 1.0, 1.0], dtype = float)
    y = numpy.array([1.0, 0.0, 0.0], dtype = float)
    assert dominance_relationship(x, y) == 1
    x = numpy.array([1.0, 1.0, 1.0], dtype = float)
    y = numpy.array([0.0, 1.0, 0.0], dtype = float)
    assert dominance_relationship(x, y) == 1
    x = numpy.array([1.0, 1.0, 1.0], dtype = float)
    y = numpy.array([0.0, 0.0, 1.0], dtype = float)
    assert dominance_relationship(x, y) == 1

# def test_maximizing_dominance_relationship_1d():
#     # x dominates y
#     x = numpy.array([0.0], dtype = float)
#     y = numpy.array([1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)

#     # x is nondominated by y
#     x = numpy.array([0.0], dtype = float)
#     y = numpy.array([0.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)

#     # x is dominated by y
#     x = numpy.array([1.0], dtype = float)
#     y = numpy.array([0.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)

# def test_maximizing_dominance_relationship_2d():
#     # x dominates y
#     x = numpy.array([0.0, 0.0], dtype = float)
#     y = numpy.array([1.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([1.0, 0.0], dtype = float)
#     y = numpy.array([1.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)

#     # x is nondominated by y
#     x = numpy.array([0.0, 0.0], dtype = float)
#     y = numpy.array([0.0, 0.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([1.0, 0.0], dtype = float)
#     y = numpy.array([0.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([0.0, 1.0], dtype = float)
#     y = numpy.array([1.0, 0.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)

#     # x is dominated by y
#     x = numpy.array([1.0, 1.0], dtype = float)
#     y = numpy.array([0.0, 0.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([1.0, 1.0], dtype = float)
#     y = numpy.array([0.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)

# def test_maximizing_dominance_relationship_3d():
#     # x dominates y
#     x = numpy.array([0.0, 0.0, 0.0], dtype = float)
#     y = numpy.array([1.0, 1.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([1.0, 0.0, 0.0], dtype = float)
#     y = numpy.array([1.0, 1.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([0.0, 1.0, 0.0], dtype = float)
#     y = numpy.array([1.0, 1.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([0.0, 0.0, 1.0], dtype = float)
#     y = numpy.array([1.0, 1.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)

#     # x is nondominated by y
#     x = numpy.array([1.0, 0.0, 0.0], dtype = float)
#     y = numpy.array([0.0, 1.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([0.0, 1.0, 0.0], dtype = float)
#     y = numpy.array([1.0, 0.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([0.0, 0.0, 1.0], dtype = float)
#     y = numpy.array([1.0, 1.0, 0.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)

#     # x is dominated by y
#     x = numpy.array([1.0, 1.0, 1.0], dtype = float)
#     y = numpy.array([0.0, 0.0, 0.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([1.0, 1.0, 1.0], dtype = float)
#     y = numpy.array([1.0, 0.0, 0.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([1.0, 1.0, 1.0], dtype = float)
#     y = numpy.array([0.0, 1.0, 0.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
#     x = numpy.array([1.0, 1.0, 1.0], dtype = float)
#     y = numpy.array([0.0, 0.0, 1.0], dtype = float)
#     assert maximizing_dominance_relationship(x, y) == minimizing_dominance_relationship(-x, -y)
