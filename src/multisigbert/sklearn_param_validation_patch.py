# from numbers import Real, Integral

class Interval:
    def __init__(self, dtype, left, right, closed="neither"):
        self.dtype = dtype
        self.left = left
        self.right = right
        self.closed = closed

class StrOptions:
    def __init__(self, options):
        self.options = set(options)