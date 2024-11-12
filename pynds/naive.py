import numpy
from pynds.relation import dominance_relationship

def krange(start: int, stop: int, skip: int):
    yield from range(start, skip)
    yield from range(skip+1, stop)

def ndsort_naive(x: numpy.ndarray, rank: numpy.ndarray, inplace: bool = True) -> numpy.ndarray:
    """
    Non-dominated sorting using the naive algorithm.

    Computational complexity: O(MN^3)
    Space complexity: O(N) (for bookkeeping)

    Parameters
    ----------
    x : numpy.ndarray
        Input matrix of shape ``(nindiv,nobj)`` to non-dominated sort.
        If ``inplace == bool``, then this matrix is modified in-place.
    rank : numpy.ndarray
        Pointer to output rank assignment array.
    inplace : bool
        Whether to modify the matrix in-place or return a modified copy.
    
    Returns
    -------
    out : numpy.ndarray
        Output matrix non-dominated sorted matrix.
    """
    # define output
    out = x if inplace else x.copy()

    # get the number of individuals and objectives
    nindiv = out.shape[0]

    # starting index
    start = 0

    # starting rank
    current_rank = 0

    # while 
    while start < nindiv:
        bookkeeping = []
        # for each individual
        for indiv in range(start, nindiv):
            nondominated = True
            # for each other individual
            for other in krange(start, nindiv, indiv):
                # if individual is dominated by other individual, mark as dominated and break
                if dominance_relationship(out[indiv], out[other]) > 0:
                    nondominated = False
                    break
            # if individual is nondominated, store index in bookkeeping
            if nondominated:
                bookkeeping.append(indiv)
        # for each index found in the bookkeeping, swap and store rank
        for ix in bookkeeping:
            out[[start,ix],:] = out[[ix,start],:]
            rank[start] = current_rank
            start += 1
        # increment rank
        current_rank += 1
    return out
