import numpy as np

def grand_index(U, V, adjusted=False):
    """
    Grand Index (13GRI) and Adjusted Grand Index (13AGRI) implementations.
    
    These measures are capable of comparing exclusive hard, fuzzy/probabilistic,
    non-exclusive hard, and possibilistic clusterings.
    
    Reference
    ---------
    Horta, D., and Campello, R. J. G. B. (2015). Comparing Hard and Overlapping Clusterings.
    Journal of Machine Learning Research, 16: 2949-2997.
    
    Parameters
    ----------
    U : numpy.ndarray
        k x n matrix representing a clustering solution with k clusters on a dataset
        with n objects. U[r, i] expresses the membership degree of the object i to
        the rth cluster.
    
    V : numpy.ndarray
        l x n matrix representing a clustering solution with l clusters on a dataset
        with n objects. V[r, i] expresses the membership degree of the object i to
        the rth cluster.
    
    adjusted : bool, default: False
        If True, calculates 13AGRI. Otherwise, calculates 13GRI.
    
    Returns
    -------
    gri : float
        Similarity score of U and V. A score of 1 represents a perfect match.
        If adjusted is True, random solutions have a score close to 0.
    """
    Ju, Su, Tu = _calculate_information_arrays(U)
    Jv, Sv, Tv = _calculate_information_arrays(V)
    Tmax = max(np.sum(Tu), np.sum(Tv))
    gri = _calculate_gri(Ju, Su, Jv, Sv, Tmax)

    if not adjusted:
        return gri

    gri_expectation = _calculate_gri_expectation(Ju, Su, Jv, Sv, Tmax)
    return (gri - gri_expectation) / (1.0 - gri_expectation)

def _calculate_information_arrays(U):
    ku, n = U.shape
    i = np.triu_indices(n, 1)
    Ju = np.dot(U.T, U)[i]
    Su = np.dot(U.T, np.dot(np.ones((ku, ku)) - np.identity(ku), U))[i]
    return Ju, Su, Ju + Su

def _calculate_gri(Ju, Su, Jv, Sv, Tmax):
    a = np.sum(np.minimum(Ju, Jv))
    d = np.sum(np.minimum(Su, Sv))
    return (a + d) / Tmax

def _calculate_gri_expectation(Ju, Su, Jv, Sv, Tmax):
    a_expectation = _calculate_expectation(Ju, Jv)
    d_expectation = _calculate_expectation(Su, Sv)
    return (a_expectation + d_expectation) / Tmax

def _calculate_expectation(Ju, Jv):
    x, y = np.sort(Ju), np.sort(Jv)
    expectation = 0.0
    
    for n in reversed(x):
        count = np.count_nonzero(n <= y)
        expectation += count * n
    
    for n in reversed(y):
        count = np.count_nonzero(n < x)
        expectation += count * n
    
    return expectation / len(Ju)
