import numpy as np
from pinball_env import PinballEnv, PinballTileCoder, available_configs


def get_env_and_coder(
        config: str = 'pinball_simple_single',
        num_tilings: int = 32,
        num_tiles: int = 8,
        iht_size: int = 4096
    ):
    """
    Returns an instance of the Pinball environment and a tile coder.

    :return: Tuple containing the Pinball environment and the tile coder.
    """
    # env
    env = PinballEnv(available_configs.get(config))
    # coder
    coder = PinballTileCoder(num_tilings, num_tiles, iht_size)

    return env, coder


def learn(
        env: PinballEnv,
        coder: PinballTileCoder,
        gamma: float =0.99,
        alpha: float =.1,
        alpha_decay: float =1.,
        min_alpha: float =.1,
        epsilon: float =.1,
        epsilon_decay: float =1.,
        min_epsilon: float =.1,
        number_of_steps: int =100_000
    ):

    def policy(Qw, state_features, epsilon):
        num_actions = Qw.shape[0]

        # Exploration
        if np.random.random() < epsilon:
            return np.random.choice(range(num_actions))

        # Compute Q values
        qs = [Qw[a] @ state_features for a in range(num_actions)]

        # Handle if there is more with same max value
        available_qs = [i for i,q in enumerate(qs) if max(qs) == q]
        return np.random.choice(available_qs)

    num_actions = env.action_space.n

    # initialize Q parameters: those will be learned
    Qw = np.zeros((num_actions, coder.iht_size))

    # reset the env
    state, info = env.reset()

    # run
    done = False
    for step in range(number_of_steps):

        # Episode done, reset
        if done:
            # reset the env
            state, info = env.reset()
            done = False

        # Construct state features using tile coding
        state_features = coder.get_state_features(*state)

        # Choose action
        action = policy(Qw, state_features, epsilon)

        # Step
        next_state, reward, done, info, _ = env.step(action)

        # Construct next state features
        next_state_features = coder.get_state_features(*next_state)

        # Compute state and next_state value
        sQ = Qw[action] @ state_features

        # Update weights
        if done:
            Qw[action] += alpha * (reward - sQ) * state_features
        else:
            nsQ = max([Qw[a] @ next_state_features for a in range(num_actions)])
            Qw[action] += alpha * (reward + gamma * nsQ - sQ) * state_features

        # End step
        alpha *= alpha_decay if alpha * alpha_decay > min_alpha else alpha
        epsilon *= epsilon_decay if epsilon * epsilon_decay > min_epsilon else epsilon
        state = next_state

    # Return learned Q function weights
    return Qw


def test(
        env: PinballEnv,
        coder: PinballTileCoder,
        Qw: np.array,
        epsilon: float =.1,
        epsilon_decay: float =1.,
        min_epsilon: float =.1,
    ):

    def policy(Qw, state_features, epsilon):
        num_actions = Qw.shape[0]

        # Exploration
        if np.random.random() < epsilon:
            return np.random.choice(range(num_actions))

        # Compute Q values
        qs = [Qw[a] @ state_features for a in range(num_actions)]

        # Handle if there is more with same max value
        available_qs = [i for i,q in enumerate(qs) if max(qs) == q]
        return np.random.choice(available_qs)

    G = 0
    steps = 0

    state, info = env.reset()
    done = False
    while not done:
        state_features = coder.get_state_features(*state)
        a = policy(Qw, state_features, epsilon)
        next_state, reward, done, info, _ = env.step(a)
        epsilon *= epsilon_decay if epsilon * epsilon_decay > min_epsilon else epsilon
        G += reward
        steps += 1
        state = next_state

    return G, steps


if __name__ == "__main__":
    # Get environment and tile coder
    env, coder = get_env_and_coder()

    # Learn Q function weights
    for number_of_steps in [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]:
        Qw = learn(env, coder, number_of_steps=number_of_steps)
        result = test(env, coder, Qw)
        print(f"Total reward from testing learned policy after {number_of_steps} steps: {result[0]} in {result[1]} steps.")
