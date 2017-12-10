from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import psutil
import sys
import datetime
from absl import flags

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import environment
from pysc2.env import sc2_env


# Initialization, covering the base class setup method
# Construct a strategy network
# Construct training methods
# Training Strategy Network
# According to the observation of the output action, covering the base class step method


class DeepQAgent(base_agent.BaseAgent):
    """A random agent for starcraft."""

    def __init__(self):
        super(DeepQAgent, self).__init__()

    def step(self, obs):
        super(DeepQAgent, self).step(obs)
        function_id = np.random.choice(obs.observation["available_actions"])
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        print(self.steps)
        print(self.reward)
        return actions.FunctionCall(function_id, args)
