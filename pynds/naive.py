import numpy
from pynds.relation import dominance_relationship
from pynds.relation import dominance_relationship_matrix

def krange(start: int, stop: int, skip: int):
    yield from range(start, skip)
    yield from range(skip+1, stop)

def ndsort_naive1(x: numpy.ndarray, front: numpy.ndarray, inplace: bool = True) -> numpy.ndarray:
    """
    Non-dominated sorting using the naive algorithm.

    Computational complexity: O(MN^3)
    Space complexity: O(N) (for bookkeeping)

    Parameters
    ----------
    x : numpy.ndarray
        Input matrix of shape ``(nindiv,nobj)`` to non-dominated sort.
        If ``inplace == bool``, then this matrix is modified in-place.
    front : numpy.ndarray
        Pointer to output front assignment array.
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

    # starting front
    current_front = 0

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
        # for each index found in the bookkeeping, swap and store front
        for ix in bookkeeping:
            out[[start,ix],:] = out[[ix,start],:]
            front[start] = current_front
            start += 1
        # increment front
        current_front += 1
    return out

def ndsort_naive2(x: numpy.ndarray, front: numpy.ndarray, dom: numpy.ndarray, rem: numpy.ndarray, mask: numpy.ndarray) -> None:
    """
    In-place non-dominated sorting using the naive algorithm, variant 2.

    Computational complexity: O(MN^3 + MN^2 + MNlogN)
        O(MN^3) for naively traversing dominance relationship matirx.
        O(MN^2) for calculating dominance relationship matrix
        O(MNlogN) for final quicksorting based on dominance and objectives

    Space complexity: O(N^2 + 2N)
        O(N^2) for storing dominance relationship matrix
        O(N) for storing remaining indices left to sort
        O(N) for storing mask of values which were nondominated

    Size definitions:
        M = number of objectives.
        N = number of individuals.

    This Python code is meant to be close to what would be found in C, so that it can be easily translated.

    Parameters
    ----------
    x : numpy.ndarray
        A matrix of shape ``(M,N)`` containing objective values for individuals.
    front : numpy.ndarray
        A vector of shape ``(N,)`` containing front assignments.
    dom : numpy.ndarray
        A matrix of shape ``(N,N)``.
        Workspace holding pairwise dominance relationships.
    rem : numpy.ndarray
        A vector of shape ``(N,)``
        Workspace holding remaining indices to visit
    mask : numpy.ndarray
        A vector of shape ``(N,)``
        Workspace holding indices marked as nondominated
    """
    # get the number of individuals
    nindiv = x.shape[0]
    nobj = x.shape[1]

    # calculate pairwise dominance relationships
    dominance_relationship_matrix(x, dom)

    # initialize number of remaining individuals, indices for remaining individuals, nondominated mask
    nrem = nindiv
    for i in range(nrem):
        rem[i] = i
        mask[i] = False

    # current front counter
    nfront = 0

    # while not all individuals have been assigned front labels
    while nrem > 0:
        # make current front assignments
        for i in range(nrem):               # for each remaining index in remaining index array
            ix = rem[i]                     # get index for current individual
            nondominated = True             # mark current individual as nondominated
            for j in range(nrem):           # for each other individual in remaining index array
                jx = rem[j]                 # get index for other individual
                if dom[ix,jx] > 0:          # if individual is dominated, mark as dominated and break loop
                    nondominated = False    # set nondominated status to False
                    break
            mask[i] = nondominated          # assign nondomination status to mask array
            if nondominated:                # make front assignment for current individual
                front[ix] = nfront
        
        # shrink remaining individuals
        offset = 0                          # initialize offset to 0
        for i in range(nrem):               # for each element in mask
            if mask[i] == True:             # if index marked as nondominated (in current front)
                offset += 1                 # increment offset for copying
                nrem -= 1                   # decrement number of remaining individuals
                continue                    # got to next element
            rem[i-offset] = rem[i]          # shift values by offset

        # increment front count
        nfront += 1

    # quicksort first by front, then by column 0, then by column 1, ...
    # use numpy because I'm lazy; would need to implement this in C.

    # get keys (must be in reverse order)
    keys = tuple(x[:,i] for i in range(nobj))[::-1] + (front,)

    # indirect sort indices for reordering
    ix = numpy.lexsort(keys)
    
    # reorder objective matrix and front matrix
    x[:,:] = x[ix,:]
    front[:] = front[ix]
