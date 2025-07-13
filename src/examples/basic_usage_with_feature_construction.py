import numpy as np
from pinball_env.pinball import PinballEnv
from pinball_env.feature_construction.tile_coder import PinballTileCoder

def random_agent_with_feature_construction(env: PinballEnv, coder: PinballTileCoder):
    """ A random policy Agent that selects actions uniformly at random. """
    behavioral_policy = np.ones(env.action_space.n) / env.action_space.n
    s, info = env.reset()
    done = False
    while not done:
        action = np.random.choice(env.action_space.n, p=behavioral_policy)
        ns, reward, done, info, _ = env.step(action)
        # Get the state features using the tile coder
        s_features = coder.get_state_features(*s)
        ns_features = coder.get_state_features(*ns)
        print(f"State features: {s_features}, Action: {action}, Next state features: {ns_features}, Reward: {reward}, Done: {done}, Info: {info}")
        s = ns


if __name__ == "__main__":
    # $ python3 random_agent.py ~/pinball_env/configs/pinball_simple_single.cfg 32 8 4096
    import argparse
    parser = argparse.ArgumentParser(description='Pinball domain')
    parser.add_argument('configuration', help='the configuration file')
    parser.add_argument('num_tilings', type=int, help='the number of tilings')
    parser.add_argument('num_tiles', type=int, help='the number of tiles per tiling')
    parser.add_argument('iht_size', type=int, help='the configuration file')
    args = parser.parse_args()
    env = PinballEnv(configuration_file=args.configuration)
    coder = PinballTileCoder(
        num_tilings=args.num_tilings,
        num_tiles=args.num_tiles,
        iht_size=args.iht_size
    )
    random_agent_with_feature_construction(env, coder)
    env.close()
