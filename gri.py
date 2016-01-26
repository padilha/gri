import numpy as np

def grand_index(U, V, adjusted=False):
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
