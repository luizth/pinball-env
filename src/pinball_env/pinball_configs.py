
pinball_hard_single = """ball 0.015
target 0.5 0.06 0.04
start 0.055 0.95

polygon 0.0 0.0 0.0 0.01 1.0 0.01 1.0 0.0
polygon 0.0 0.0 0.01 0.0 0.01 1.0 0.0 1.0
polygon 0.0 1.0 0.0 0.99 1.0 0.99 1.0 1.0
polygon 1.0 1.0 0.99 1.0 0.99 0.0 1.0 0.0
polygon 0.034 0.852 0.106 0.708 0.33199999999999996 0.674 0.17599999999999996 0.618 0.028 0.718
polygon 0.15 0.7559999999999999 0.142 0.93 0.232 0.894 0.238 0.99 0.498 0.722
polygon 0.8079999999999999 0.91 0.904 0.784 0.7799999999999999 0.572 0.942 0.562 0.952 0.82 0.874 0.934
polygon 0.768 0.814 0.692 0.548 0.594 0.47 0.606 0.804 0.648 0.626
polygon 0.22799999999999998 0.5760000000000001 0.39 0.322 0.3400000000000001 0.31400000000000006 0.184 0.456
polygon 0.09 0.228 0.242 0.076 0.106 0.03 0.022 0.178
polygon 0.11 0.278 0.24600000000000002 0.262 0.108 0.454 0.16 0.566 0.064 0.626 0.016 0.438
polygon 0.772 0.1 0.71 0.20599999999999996 0.77 0.322 0.894 0.09600000000000002 0.8039999999999999 0.17600000000000002
polygon 0.698 0.476 0.984 0.27199999999999996 0.908 0.512
polygon 0.45 0.39199999999999996 0.614 0.25799999999999995 0.7340000000000001 0.438
polygon 0.476 0.868 0.552 0.8119999999999999 0.62 0.902 0.626 0.972 0.49 0.958
polygon 0.61 0.014000000000000002 0.58 0.094 0.774 0.05000000000000001 0.63 0.054000000000000006
polygon 0.33399999999999996 0.014 0.27799999999999997 0.03799999999999998 0.368 0.254 0.7 0.20000000000000004 0.764 0.108 0.526 0.158
polygon 0.294 0.584 0.478 0.626 0.482 0.574 0.324 0.434 0.35 0.39 0.572 0.52 0.588 0.722 0.456 0.668
"""

pinball_simple_single = """ball 0.02
target 0.9 0.2 0.04
start 0.2 0.9

polygon 0.0 0.0 0.0 0.01 1.0 0.01 1.0 0.0
polygon 0.0 0.0 0.01 0.0 0.01 1.0 0.0 1.0
polygon 0.0 1.0 0.0 0.99 1.0 0.99 1.0 1.0
polygon 1.0 1.0 0.99 1.0 0.99 0.0 1.0 0.0

polygon 0.35 0.4 0.45 0.55 0.43 0.65 0.3 0.7 0.45 0.7 0.5 0.6 0.45 0.35
polygon 0.2 0.6 0.25 0.55 0.15 0.5 0.15 0.45 0.2 0.3 0.12 0.27 0.075 0.35 0.09 0.55
polygon 0.3 0.8 0.6 0.75 0.8 0.8 0.8 0.9 0.6 0.85 0.3 0.9
polygon 0.8 0.7 0.975 0.65 0.75 0.5 0.9 0.3 0.7 0.35 0.63 0.65
polygon 0.6 0.25 0.3 0.07 0.15 0.175 0.15 0.2 0.3 0.175 0.6 0.3
polygon 0.75 0.025 0.8 0.24 0.725 0.27 0.7 0.025
"""


# Dictionary to hold available configurations
import random
class ConfigDict(dict):
    """A dictionary that supports sampling random values."""

    def sample(self):
        """Return a random configuration from the available configs."""
        if not self:
            raise ValueError("No configurations available to sample from")
        return random.choice(list(self.values()))

    def __repr__(self):
        """Return a string representation of the dictionary."""
        return self.keys().__repr__()

# Dictionary to hold available configurations
available_configs = ConfigDict({
    "pinball_hard_single": pinball_hard_single,
    "pinball_simple_single": pinball_simple_single,
})


if __name__ == "__main__":
    # Example usage
    print("Available Configurations:", available_configs)
    config = available_configs.sample()
    print("Sampled Config:\n", config)
