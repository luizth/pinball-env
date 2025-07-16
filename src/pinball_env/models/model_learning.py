import numpy as np
from typing import Optional, Tuple
from pinball_env import PinballEnv, PinballTileCoder, available_configs

from pinball_env.external.training_logger import TrainingLogger


def rmse(true: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between true and predicted values.

    Args:
        true (np.ndarray): The true values.
        predicted (np.ndarray): The predicted values.

    Returns:
        float: The RMSE value.
    """
    return np.sqrt(np.mean((true - predicted) ** 2))


def save_model(model: np.ndarray, file_path: str = "model.npy") -> None:
    """
    Save the learned model to a file.

    Args:
        model (np.ndarray): The learned model to save.
        file_path (str): The path where the model will be saved.

    Returns:
        None
    """
    np.save(file_path, model)
    print(f"Model saved to {file_path}")


def get_env_and_coder(
        env_configuration: str = "pinball_simple_single",
        num_tilings: int = 32,
        num_tiles: int = 8,
        iht_size: int = 4096
    ) -> Tuple[PinballEnv, PinballTileCoder]:
    """
    Initialize the Pinball environment and tile coder with default configurations.

    Returns:
        PinballEnv: The initialized pinball environment.
        PinballTileCoder: The initialized tile coder for the environment.
    """
    env = PinballEnv(available_configs.get(env_configuration, None))
    tile_coder = PinballTileCoder(num_tilings, num_tiles, iht_size)
    return env, tile_coder


def sample_states_and_next_states(
        env: PinballEnv,
        tile_coder: PinballTileCoder,
        num_samples: int = 1000
    ) -> np.ndarray:
    """
    Sample states from the Pinball environment.

    Args:
        env (PinballEnv): The pinball environment.
        num_samples (int): Number of states to sample.

    Returns:
        np.ndarray: Array of sampled states.
    """
    _env = env.copy()
    s_and_ns = []
    for _ in range(num_samples):
        state = _env.observation_space.sample()
        _env.set_state(state)
        next_state, _, _, _, _ = _env.step(4)  # do nothing
        state_features = tile_coder.get_state_features(*state)
        next_state_features = tile_coder.get_state_features(*next_state)
        s_and_ns.append((state_features,next_state_features))
    return np.array(s_and_ns)


def learn_model(
        env: PinballEnv,
        tile_coder: PinballTileCoder,
        alpha: float = 0.1,
        gamma: float = 0.99,
        number_of_steps: int = 10_000,
        log_freq: int = 1,
        logger: Optional[TrainingLogger] = None
    ):
    """
    Learn a transition model of the Pinball environment for a provided tile coder.

    Args:
        env (PinballEnv): The pinball environment.
        tile_coder (PinballTileCoder): The tile coder for the environment.
        number_of_steps (int): Number of steps to run for learning.

    Returns:
        None
    """

    # Initialize model parameters
    model_dim = tile_coder.iht_size
    W = np.zeros((model_dim, model_dim))
    # We = np.zeros((model_dim, model_dim))

    transition_errors = np.zeros(number_of_steps)

    # Initialize policy
    action_dim = env.action_space.n
    behavior_policy_probs = np.ones(action_dim) / action_dim

    # Learning loop
    state, info = env.reset()
    state = env.observation_space.sample()  # Reset state to a random sample
    env.set_state(state)
    done = False
    for step in range(number_of_steps):

        # If episode done
        if done:
            state, info = env.reset()
            state = env.observation_space.sample()  # Reset state to a random sample
            env.set_state(state)
            done = False

            # We = np.zeros((model_dim, model_dim))

        # Get action from behavior policy
        # action = np.random.choice(action_dim, p=behavior_policy_probs)

        # Set random state
        state = env.observation_space.sample()  # Reset state to a random sample
        env.set_state(state)

        # Take action in the environment
        next_state, reward, done, truncated, info = env.step(4)  # do nothing

        # Get state and next state features using the tile coder
        state_features = tile_coder.get_state_features(*state)
        next_state_features = tile_coder.get_state_features(*next_state)

        # Calculate the transition error for the current step
        sampled_features = sample_states_and_next_states(env, tile_coder, num_samples=100)
        transition_errors[step] = np.mean([
            rmse(ns_feat, np.dot(W, s_feat)) for s_feat, ns_feat in sampled_features
        ])

        # Update model using the tile coder features
        for j in range(model_dim):
            # Predict state features using the expectation model
            N_s = np.dot(W[j], state_features)
            N_ns = np.dot(W[j], next_state_features)

            # TD
            target = int(done) * next_state_features[j] + gamma * (1-done) * N_ns
            error = target - N_s

            # Gradient with clipping
            gradient = alpha * error * state_features
            gradient = np.clip(gradient, -1.0, 1.0)  # Prevent explosive updates

            # Update feature weights the direction of the gradient
            W[j] += gradient

            # Optional: Add small regularization to prevent weights from growing too large
            # W[j] *= 0.9999  # L2 regularization factor

        if logger is not None and step % log_freq == 0:
            # Log the transition error every log_freq steps
            logger.log_step(step=step, error=transition_errors[step])

    return W, transition_errors


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # env params
    parser.add_argument("--cfg", type=str, default="pinball_simple_single",
                        choices=available_configs.keys(),
                        help="configuration for the Pinball environment.")
    # tile coder params
    parser.add_argument("--tc_num_tilings", type=int, default=32,
                        help="tile coder number of tilings.")
    parser.add_argument("--tc_num_tiles", type=int, default=8,
                        help="tile coder number of tiles in each tiling.")
    parser.add_argument("--tc_iht_size", type=int, default=1048,
                        help="tile coder size of the index hash table.")
    # model learning params
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="learning rate for model learning.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor for model learning.")
    parser.add_argument("--number_of_steps", type=int, default=10_000,
                        help="number of steps to run for learning the model.")
    # logging params
    parser.add_argument("--stream_log", action='store_true', default=False,
                        help="whether to log the transition error to the console.")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="frequency of logging the transition error.")
    parser.add_argument("--save", action='store_true', default=False,
                        help="whether to save the learned model to a file.")

    # Parse arguments
    args = parser.parse_args()

    # Experiment name
    exp = f"ModelLearning_Env{args.cfg}_TileCoder{args.tc_iht_size}_Steps{args.number_of_steps}_alpha{args.alpha}"

    # Initialize logger if logging to stream
    logger = None
    if args.stream_log:
        logger = TrainingLogger(
            exp,
            log_dir="/home/luizalfredo/experimentslog",
            number_of_steps=args.number_of_steps
        )

    # Get environment and tile coder
    env, tile_coder = get_env_and_coder(
        env_configuration=args.cfg,
        num_tilings=args.tc_num_tilings,
        num_tiles=args.tc_num_tiles,
        iht_size=args.tc_iht_size
    )

    # Learn the model
    W, transition_errors = learn_model(
        env,
        tile_coder,
        alpha=args.alpha,
        gamma=args.gamma,
        number_of_steps=args.number_of_steps,
        log_freq=args.log_freq,
        logger=logger
    )

    # Save the model if requested
    if args.save:
        from pathlib import Path
        path = str(Path(__file__).parent)
        filepath =  path + f"/model_{exp}.npy"
        save_model(W, filepath)
        save_model(transition_errors, filepath.replace("model_", "errors_"))
