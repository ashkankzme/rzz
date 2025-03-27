from mpmath import mp

def zeros_starting_at_N(N, number_of_zeros):
    """
    Returns a list of tuples (index, rzz) where rzz is an mp.mpf.
    For testing, we simulate zeros.
    """
    results = []
    for i in range(N, N + number_of_zeros):
        # Simulate a zero by a simple function (in reality, these would be precomputed high-precision values)
        simulated_zero = mp.mpf("50.0000") + mp.mpf("0.01") * (i - N)
        results.append((i, simulated_zero))
    return results
