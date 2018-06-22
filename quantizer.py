import random


class Quantizer:
    """
    quantize value range.
    if we want to quantize -2 to 2 into 4 bins, -2 to -1 | -1 to 0 | 0 to 1 | 1 to 2
    each bin range should be 1.
    also index of 0.5 should return 2.

    Examples
    Note: some value may return incorrect result because of float imprecision
    >>> q = Quantizer(-2, 2, 4)
    >>> q.index(-1.9)
    0
    >>> q.round(-1.9)
    -2.0
    >>> q.index(0.5)
    2
    >>> q.round(0.5)
    1.0

    >>> q = Quantizer(-0.5, 1, 15)
    >>> q.index(-0.38)
    1
    >>> q.round(-0.38)
    -0.4
    >>> q.index(0.18)
    6

    >> q.round(0.18)
    This one have float imprecision problem for now
    """
    def __init__(self, start, end, bins):
        self.start = start
        self.end = end
        self.bins = bins
        # move start to 0.0 and end so that calculating index is easier (not use for now)
        self.moved_start = 0.0
        self.moved_end = abs(end - start)
        # range of bin
        self.step = abs(end - start) / bins

    def index(self, value):
        """
        return which index(bin) value belong to
        """
        if value < self.start or value > self.end:
            raise IndexError
        else:
            moved_value = abs(value - self.start)
            return int(moved_value // self.step)

    def round(self, value):
        """
        round to nearest bin value
        """
        # FIXME: this is having float imprecision problem
        if value < self.start or value > self.end:
            random_index = random.choice(range(self.bins))
            v_random = self.start + self.step * random_index
            return v_random
        else:
            idx = self.index(value)
            v1 = self.start + self.step * idx
            v2 = v1 + self.step
            a = abs(value - v1)
            b = abs(value - v2)
            if a >= b:
                return float(v2)
            else:
                return float(v1)

