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

import Qnetwork

step_mul = 8
FLAGS = flags.FLAGS

flags.DEFINE_string("map_name", "MoveToBeacon")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%m%d%H%M")

def main():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(
        map_name= "MoveToBeacon",
        step_mul= step_mul,
        visualize= True,
        screen_size_px=(16,16),
        minimap_size_px=(16,16)) as env:





