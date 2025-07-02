import numpy as np
from pinball_env.pinball import PinballEnv


def random_agent(env: PinballEnv):
    """ A random policy Agent that selects actions uniformly at random. """
    behavioral_policy = np.ones(env.action_space.n) / env.action_space.n
    s, info = env.reset()
    done = False
    while not done:
        action = np.random.choice(env.action_space.n, p=behavioral_policy)
        ns, reward, done, info, _ = env.step(action)
        print(f"State: {s}, Action: {action}, Next state: {ns}, Reward: {reward}, Done: {done}, Info: {info}")
        s = ns


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pinball domain')
    parser.add_argument('configuration', help='the configuration file')
    args = parser.parse_args()
    env = PinballEnv(configuration_file=args.configuration)
    random_agent(env)
    env.close()
