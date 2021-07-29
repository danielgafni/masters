from typing import Optional

import gym


def make_wrapped_env(env_name: str, random_seed: Optional[int] = None):
    env = gym.make(env_name)
    env.seed(random_seed)
    return env
