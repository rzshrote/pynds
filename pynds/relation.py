import numpy

def minimizing_dominance_relationship(x: numpy.ndarray, y: numpy.ndarray) -> int:
    """
    Determine the dominance relationship between two vectors.
    Assumes that objectives are to be minimized.

    Parameters
    ----------
    x : numpy.ndarray
        First vector for which to determine the dominance relationship.
    y : numpy.ndarray
        Second vector for which to determine the dominance relationship.
    
    Returns
    -------
    out : int
        If x dominates y, then return -1.
            (x < y) == -1
        If x is non-dominated by y, then return 0.
            (x !< y and y !< x) == 0
        If x is dominated by y, then return 1.
            (x > y) == 1
    """
    x_le_y = numpy.all(x <= y)
    y_le_x = numpy.all(x >= y)
    x_lt_y = numpy.any(x < y)
    y_lt_x = numpy.any(x > y)

    x_dom_y = x_le_y and x_lt_y
    y_dom_x = y_le_x and y_lt_x

    if x_dom_y:
        return -1
    elif y_dom_x:
        return 1
    else:
        return 0

def maximizing_dominance_relationship(x: numpy.ndarray, y: numpy.ndarray) -> int:
    """
    Determine the dominance relationship between two vectors.
    Assumes that objectives are to be maximize.

    Parameters
    ----------
    x : numpy.ndarray
        First vector for which to determine the dominance relationship.
    y : numpy.ndarray
        Second vector for which to determine the dominance relationship.
    
    Returns
    -------
    out : int
        If x dominates y, then return -1.
            (x < y) == -1
        If x is non-dominated by y, then return 0.
            (x !< y and y !< x) == 0
        If x is dominated by y, then return 1.
            (x > y) == 1
    """
    x_ge_y = numpy.all(x >= y)
    y_ge_x = numpy.all(x <= y)
    x_gt_y = numpy.any(x > y)
    y_gt_x = numpy.any(x < y)

    x_dom_y = x_ge_y and x_gt_y
    y_dom_x = y_ge_x and y_gt_x

    if x_dom_y:
        return -1
    elif y_dom_x:
        return 1
    else:
        return 0
