from Qnetwork import *
from util_func import *
from experience_buffer import *
from pysc2.env import sc2_env
from pysc2 import maps
from pysc2.lib import actions
import os
import sys
import gflags as flags
import pysc2.lib.features
import time

FLAGS = flags.FLAGS
_UNIT_TYPE_VIEW = features.SCREEN_FEATURES.unit_type.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_TERRAN_BARRACKS = 21
_SELECT_UNIT = 5
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_BUILD_MARINE = 477
_ALL_OF_TYPE_SELECT_POINT = 2
_SELECT_SELECT_POINT = 0
_TOGGLE_SELECT_POINT = 1
_ADDLLTYPE_SELECT_POINT = 3

def main():
    FLAGS(sys.argv)
    maps.get(map_name)  # Assert the map exists.

    tf.reset_default_graph()
    mainQN = Qnetwork(h_size)
    targetQN = Qnetwork(h_size)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables, tau)

    myBuffer = experience_buffer()

    with sc2_env.SC2Env(map_name,
                        step_mul=step_mul,
                        game_steps_per_episode=0,  # maxEpLength?
                        screen_size_px=(screen_xy, screen_xy),
                        minimap_size_px=(64, 64),
                        visualize=True) as env:
        saver.restore(path + '/model-5000.ckpt')
        print("Restored")
        action_spec = env.action_spec()
        # Reset environment and get first new observation
        timestep = env.reset()[0]
        s = processState(timestep.observation["screen"][_UNIT_TYPE_VIEW])




if __name__ == "__main__":
    main()