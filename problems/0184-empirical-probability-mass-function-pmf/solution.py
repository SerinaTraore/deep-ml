from collections import Counter
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    samples_list = list(samples)
    total = len(samples_list)

    counts = Counter(samples_list)
    pmf = [(val, count / total) for val, count in sorted(counts.items())]
    return pmf
    pass