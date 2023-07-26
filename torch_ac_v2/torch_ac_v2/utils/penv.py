import multiprocessing
import gymnasium as gym
import math
import numpy

multiprocessing.set_start_method("fork",force=True)

def worker(conn, env):
    """
    The worker class interacts with each environment individually and sends the result of the interaction
    """
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            # no agent position since this is the built-in OpenAI method
            obs, reward, terminated, truncated,  info = env.step(data)
            agent_loc = env.agent_pos
            if terminated or truncated:
                obs, _ = env.reset()
                agent_loc = env.agent_pos
            conn.send((obs, reward, terminated, truncated,agent_loc, info))
        elif cmd == "reset":
            obs, _ = env.reset()
            agent_loc = env.agent_pos
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes.
    The ParallelEnv class manages the workers and controls the interactions of multiple workers at the same time
    """

    def __init__(self, envs,wrapper = None,beta=None):
        assert len(envs) >= 1, "No environment given."

        self.beta = beta

        if wrapper is not None:
            self.envs = [wrapper(env,self.beta) if wrapper is not None else env for env in envs]
        else:
            self.envs = envs

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = multiprocessing.Pipe()
            self.locals.append(local)
            p = multiprocessing.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()[0]] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, terminated, truncated, info = self.envs[0].step(actions[0])
        agent_loc = self.envs[0].agent_pos
        if terminated or truncated:
            obs, _ = self.envs[0].reset()
            agent_loc = self.envs[0].agent_pos
        results = zip(*[(obs, reward, terminated, truncated, agent_loc, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError
    
    
class ActionBonus_v2(gym.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ActionBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = ActionBonus(env)
        >>> _, _ = env_bonus.reset(seed=0)
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
    """

    def __init__(self, env, beta=0.005):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.counts = {}
        self.beta = beta

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.unwrapped
        obs_flat = obs['image'].flatten()
        # tup = (tuple(obs_flat), env.agent_dir, action)
        tup = (tuple(obs_flat), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = self.beta / math.sqrt(new_count)
        reward +=  bonus

        return obs, reward, terminated, truncated, info


class PositionBonus(gym.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    Note:
        This wrapper was previously called ``StateBonus``.
    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import PositionBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = PositionBonus(env)
        >>> obs, _ = env_bonus.reset(seed=0)
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        0.7071067811865475
    """

    def __init__(self, env, beta=1.0):
        """A wrapper that adds an exploration bonus to less visited positions.
        Args:
            env: The environment to apply the wrapper
            beta: The coefficient for the bonus (default: 1.0)
        """
        super().__init__(env)
        self.counts = {}
        self.beta = beta

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = self.beta / math.sqrt(new_count)
        reward +=  bonus

        return obs, reward, terminated, truncated, info