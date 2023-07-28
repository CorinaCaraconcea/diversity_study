import gymnasium as gym


def make_env(env_key, seed=None, render_mode=None):
    print(env_key)
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env

