import numpy

def dominance_relationship(x: numpy.ndarray, y: numpy.ndarray) -> int:
    """
    Determine the dominance relationship between two vectors.
    Assumes that objectives are to be minimized.
    Assumes that all objective values are numbers or infinity.
    NaN values will not be handled correctly.

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

def dominance_relationship_matrix(x: numpy.ndarray, dom: numpy.ndarray) -> None:
    """
    Calculate the dominance relationships between all rows in a matrix.

    Parameters
    ----------
    x : numpy.ndarray
        A matrix of shape ``(nindiv,nobj)`` containing objective function values.
        Input matrix.
    dom : numpy.ndarray
        A matrix of shape ``(nindiv,nindiv)`` to store dominance relationships.
        Output matrix.
        If x[i] dominates x[j],           then dom[i,j] == -1.
        If x[i] is non-dominated by x[j], then dom[i,j] == 0.
        If x[i] is dominated by x[j],     then dom[i,j] == 1.
    """
    # get number of individuals
    nindiv = x.shape[0]

    # get expected and observed shapes
    eshape = (nindiv,nindiv)
    dshape = dom.shape

    # test output size
    if eshape != dshape:
        raise ValueError("Output matrix ``dom`` is not the correct shape: expected ``{0}`` but received ``{1}``".format(eshape,dshape))
    
    # calculate dominance relationships
    for i in range(nindiv):
        for j in range(nindiv):
            dom[i,j] = dominance_relationship(x[i], x[j])
