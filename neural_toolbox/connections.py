from math_utils import get_value


class Connection:
    def __init__(self, in_nuran, weight=None):
        self.in_nuran = in_nuran
        self.weight = weight if weight is not None else get_value()
