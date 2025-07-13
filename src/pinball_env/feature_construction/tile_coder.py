import numpy as np
import pinball_env.external.tiles3 as tc


class PinballTileCoder:
    def __init__(self, num_tilings, num_tiles, iht_size=None):
        """
        tips:
            numTilings, should be a power of two greater or equal to four times the number of floats

        Initializes the Pinball Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the tiles are the same

        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """

        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        if iht_size is None:
            iht_size = self._find_iht_size()
        self.iht_size = iht_size
        self.iht = tc.IHT(iht_size)

    def _find_iht_size(self):
        """Calculate the size of the IHT based on the number of tilings and tiles."""

        # compute exact iht size for 4 dimensions: xpos, ypos, xvel, yvel
        exact = self.num_tilings * self.num_tiles * 4

        # find the next power of 2 greater than or equal to exact
        iht_size = 1
        while iht_size < exact:
            iht_size *= 2
        return iht_size

    def get_tiles(self, xpos, ypos, xvel, yvel):

        # Scale the inputs to fit within the tile range
            # Normalization function: (x - a) / (b - a)
        # [a, b] is the range and [x] is the input value
        # x - a: denotes how many units x is over a. if x = a, then x - a = 0, which is the LOWER BOUND
        # b - a: denotes the width of the range. if x = b, then x - a = b - a, then division is 1, which is the UPPER BOUND

        # Define the ranges for scaling
        xpos_min, xpos_max = 0., 1.
        ypos_min, ypos_max = 0., 1.
        xvel_min, xvel_max = -1., 1.
        yvel_min, yvel_max = -1., 1.

        # Scale the inputs
        xpos_scaled = (xpos - xpos_min) / (xpos_max - xpos_min) * self.num_tiles
        ypos_scaled = (ypos - ypos_min) / (ypos_max - ypos_min) * self.num_tiles
        xvel_scaled = (xvel - xvel_min) / (xvel_max - xvel_min) * self.num_tiles
        yvel_scaled = (yvel - yvel_min) / (yvel_max - yvel_min) * self.num_tiles

        # Use tiles function to get tile indices
        tile_indices = tc.tiles(self.iht, self.num_tilings, [xpos_scaled, ypos_scaled, xvel_scaled, yvel_scaled])

        return np.array(tile_indices)

    def get_state_features(self, xpos, ypos, xvel, yvel):
        """
        Get the feature vector for the given state.
        This is a binary vector where each active tile index is set to 1.
        """
        active_tiles = self.get_tiles(xpos, ypos, xvel, yvel)
        feature_vector = np.zeros(self.iht_size)
        feature_vector[active_tiles] = 1.0
        return feature_vector


if __name__ == "__main__":
    # Example usage
    num_tilings = 128
    num_tiles = 8
    tile_coder = PinballTileCoder(num_tilings, num_tiles)

    # Example state
    xpos = 0.7
    ypos = 0.2
    xvel = 0.8
    yvel = -0.1

    # Get active tiles for the given state
    active_tiles = tile_coder.get_tiles(xpos=xpos, ypos=ypos, xvel=xvel, yvel=yvel)
    print("Active tiles:", active_tiles)

    # Feature vector
    feature_vector = np.zeros(tile_coder.iht_size)
    feature_vector[active_tiles] = 1.
    print("Feature vector:", feature_vector)

    # w of value function
    w = np.zeros(tile_coder.iht_size)  # Initialize weights to zero

    # Value
    value = np.dot(w, feature_vector)
    print("Value:", value)
