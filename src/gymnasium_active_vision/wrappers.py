from gymnasium import ObservationWrapper
from gymnasium.core import Env, ObsType, ActType, WrapperObsType

import numpy as np


class RectangularFoveaWrapper(ObservationWrapper):
    """
    Wraps an environment such that only a rectangular crop of the 
    actual observation is passed to the agent. The position of this 
    center of attention can be changed by the agent on each time 
    step using a visual action.
    """

    def __init__(self, env: Env[ObsType, ActType], left: int, top: int, width: int, height: int):
        """
        Wraps a :class:`gymnasium.Env` with the :class:`gymnasium_active_vision.RectangularFoveaWrapper`.

        Parameters
        ----------
        env : :class:`gymnasium.Env`
            The environment to wrap
        left : :class:`int`
            The x-coordinate of the top left corner of the crop
        top : :class:`int`
            The y-coordinate of the top left corner of the crop
        width : :class:`int`
            The width of the crop
        height : :class:`int`
            The height of the crop
        """

        self._left = left
        self._top = top
        self._width = width
        self._height = height

        super().__init__()

    def observation(self, observation: ObsType) -> WrapperObsType:
        """
        Crops the observation
        
        Examples
        --------
        >>> import gymnasium as gym
        >>> env = gym.make("BreakoutNoFrameskip-v4)
        >>> env = RectangularFoveaWrapper(env, 0, 0, 20, 20)
        >>> obs = 255 * np.random((84, 84, 1))
        >>> new_obs = env.observation(obs)
        >>> new_obs.shape
        (20, 20, 1)

        """
        
        observation = observation[self._top:self._top + self._height, self._left: self._left + self._width]

        return super().observation(observation)
