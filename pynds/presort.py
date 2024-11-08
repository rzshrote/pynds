import numpy

def presort_matrix(x: numpy.ndarray, inplace: bool = True) -> numpy.ndarray:
    """
    Presort matrix in ascending order, starting with the first column,
    then considering subsequent columns if identical values exist.
    Assumes objectives are minimizing.

    Parameters
    ----------
    x : numpy.ndarray
        A matrix to sort of shape ``(n,m)``.
        
        Where:

        - ``n`` is the number of solutions.
        - ``m`` is the number of objectives.

    inplace : bool
        Whether to sort the matrix in-place.
    
    Returns
    -------
    out : numpy.ndarray
        A pointer to the sorted matrix.
    """
    # get each column and use as a key
    keys = tuple(x[:,i] for i in range(x.shape[1]))

    # calculate indices, but reverse keys because lexsort uses last key first
    ix = numpy.lexsort(keys[::-1])

    # copy pointer to x
    out = x

    # apply reorder
    if inplace:
        out[:,:] = x[ix,:]
    else:
        out = x[ix,:]
    
    return out