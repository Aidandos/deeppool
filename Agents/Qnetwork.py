import numpy as np
import tensorflow as tf
import os
import psutil
import sys
import datetime
from absl import flags

from pysc2.lib import actions
from pysc2.maps import mini_games
from pysc2.env import environment
from pysc2.env import sc2_env


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
