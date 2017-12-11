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

def process_observation(observation, action_spec, observation_spec):
    # reward
    reward = observation.reward
    # features
    features = observation.observation
    spatial_features = ['minimap', 'screen']
    variable_features = ['cargo', 'multi_select', 'build_queue']
    available_actions = ['available_actions']
    # the shapes of some features depend on the state (eg. shape of multi_select depends on number of units)
    # since tf requires fixed input shapes, we set a maximum size then pad the input if it falls short
    max_no = {'available_actions': len(action_spec.functions), 'cargo': 500, 'multi_select': 500, 'build_queue': 10}
    nonspatial_stack = []
    for feature_label, feature in observation.observation.items():
        if feature_label not in spatial_features + variable_features + available_actions:
            nonspatial_stack = np.concatenate((nonspatial_stack, feature.reshape(-1)))
        elif feature_label in variable_features:
            padded_feature = np.concatenate((feature.reshape(-1), np.zeros(max_no[feature_label] * observation_spec['single_select'][1] - len(feature.reshape(-1)))))
            nonspatial_stack = np.concatenate((nonspatial_stack, padded_feature))
        elif feature_label in available_actions:
            available_actions_feature = [1 if action_id in feature else 0 for action_id in np.arange(max_no['available_actions'])]
            nonspatial_stack = np.concatenate((nonspatial_stack, available_actions_feature))
    nonspatial_stack = np.expand_dims(nonspatial_stack, axis=0)
    # spatial_minimap features
    minimap_stack = np.expand_dims(np.stack(features['minimap'], axis=2), axis=0)
    # spatial_screen features
    screen_stack = np.expand_dims(np.stack(features['screen'], axis=2), axis=0)
    # is episode over?
    episode_end = observation.step_type == environment.StepType.LAST
    return reward, nonspatial_stack, minimap_stack, screen_stack, episode_end


class Qnetwork():
    def __init__(self, observation_spec, action_spec):
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
        # Tanh activation for the nonspatial features
        # dense = fully - connected layer
        self.nonspatial_dense = tf.layers.dense(
            inputs=self.inputs_nonspatial,
            units=32,
            activation=tf.tanh)

        # Convolution layer (1) for the screen features
        # 16 filters of 8 x 8 and stride 4, padding is added to match the size
        self.screen_conv1 = tf.layers.conv2d(
            inputs=self.inputs_spatial_screen,
            filters=16,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding='valid',
            activation=tf.nn.relu)
        # Convolution layer (2) for the screen features
        # 32 filters of 4 x 4 and stride 2, padding is added to match the size
        self.screen_conv2 = tf.layers.conv2d(
            inputs=self.screen_conv1,
            filters=32,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding='valid',
            activation=tf.nn.relu)
        # Convolution layer (1) for the minimap features
        # 16 filters of 8 x 8 and stride 4, padding is added to match the size
        self.minimap_conv1 = tf.layers.conv2d(
            inputs=self.inputs_spatial_minimap,
            filters=16,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding='valid',
            activation=tf.nn.relu)
        # Convolution layer (2) for the minimap features
        # 32 filters of 4 x 4 and stride 2, padding is added to match the size
        self.minimap_conv2 = tf.layers.conv2d(
            inputs=self.minimap_conv1,
            filters=32,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding='valid',
            activation=tf.nn.relu)

class DeepQAgent(base_agent.BaseAgent):
    """A random agent for starcraft."""

    def __init__(self):
        super(DeepQAgent, self).__init__()
        tf.reset_default_graph()
        self.sess = tf.Session()

    def setup(self, obs_spec, action_spec):
        super(DeepQAgent, self).setup(obs_spec, action_spec)

        gamma = .99  # Discount rate for advantage estimation and reward discounting
        load_model = False
        model_path = './model'
        map_name = "MoveToBeacon"

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.network = Qnetwork(self.obs_spec, self.action_spec)
        #saver = tf.train.Saver(max_to_keep=5)
        print("Intitializing")

        if load_model == True:
               print('Loading Model...')
               ckpt = tf.train.get_checkpoint_state(model_path)
                #saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def reset(self):
        super(DeepQAgent, self).reset()

    def step_action(self, obs):

        reward, nonspatial_stack, minimap_stack, screen_stack, episode_end = process_observation(obs,
                                                                                                 self.action_spec,
                                                                                                 self.obs_spec)
        spatial_mini = self.sess.run(
            [self.network.minimap_conv2],
            feed_dict={self.network.inputs_spatial_screen: screen_stack,
                       self.network.inputs_spatial_minimap: minimap_stack,
                       self.network.inputs_nonspatial: nonspatial_stack})

        print(spatial_mini)

    def step(self, obs):
        super(DeepQAgent, self).step(obs)

        self.step_action(obs)

        function_id = np.random.choice(obs.observation["available_actions"])
        args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)
