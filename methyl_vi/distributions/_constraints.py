from torch.distributions.constraints import Constraint


class _OpenInterval(Constraint):
    """Constrain to a real interval `(lower_bound, upper_bound)`."""

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__()

    def check(self, value):
        return (self.lower_bound < value) & (value < self.upper_bound)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += f"(lower_bound={self.lower_bound}, upper_bound={self.upper_bound})"
        return fmt_string


# Public interface
open_interval = _OpenInterval
