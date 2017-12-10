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



class Qnetwork():
    def __init__(self, action_spec, observation_spec):
        # get size of features from action_spec and observation_spec
        nonspatial_size = 0
        spatial_features = ['minimap', 'screen']
        initially_zero_features = {'cargo': 500, 'multi_select': 500, 'build_queue': 10, 'single_select': 1}
        for feature_name, feature_dim in observation_spec.items():
            if feature_name not in spatial_features:
                if feature_name == 'available_actions':
                    feature_size = len(action_spec.functions)
                elif feature_name in initially_zero_features:
                    feature_size = initially_zero_features[feature_name] * feature_dim[1]
                else:
                    feature_size = 1
                    for dim in feature_dim:
                        feature_size *= dim
                nonspatial_size += feature_size
        screen_channels = observation_spec['screen'][0]
        minimap_channels = observation_spec['minimap'][0]

        # Architecture here follows Atari-net Agent described in [1] Section 4.3
        # Intitializing Placeholders for the nonspatial features
        self.inputs_nonspatial = tf.placeholder(shape=[None, nonspatial_size], dtype=tf.float32)
        # Placeholder for screen and minimap feature layers, of (None) x width x length x number of layers
        self.inputs_spatial_screen = tf.placeholder(
            shape=[None, observation_spec['screen'][1], observation_spec['screen'][2], screen_channels],
            dtype=tf.float32)
        self.inputs_spatial_minimap = tf.placeholder(
            shape=[None, observation_spec['minimap'][1], observation_spec['minimap'][2], minimap_channels],
            dtype=tf.float32)


class DeepQAgent(base_agent.BaseAgent):
    """A random agent for starcraft."""

    def __init__(self):
        super(DeepQAgent, self).__init__()
        gamma = .99  # Discount rate for advantage estimation and reward discounting
        load_model = False
        model_path = './model'
        map_name = "MoveToBeacon"
        #assert map_name in mini_games.mini_games

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        action_spec = self.action_spec
        observation_spec = self.obs_spec

        tf.reset_default_graph()
        self.network = Qnetwork(self.action_spec,self.obs_spec)
        saver = tf.train.Saver(max_to_keep=5)

        with tf.Session() as self.sess:
            if load_model == True:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(model_path)
                saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())


    def step(self, obs):
        super(DeepQAgent, self).step(obs)
        # action_spec = self.action_spec
        # observation_spec = self.obs_spec
        #
        # feed_dict = {
        #     action_spec,
        #     observation_spec
        # }
        #
        # non_spatial, spatial_mini, spatial_screen= self.sess.run([self.network.inputs_nonspatial, self.network.inputs_spatial_minimap, self.network.inputs_spatial_screen],
        #               feed_dict=feed_dict)
        #
        # print(spatial_mini)

        function_id = np.random.choice(obs.observation["available_actions"])
        args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)
