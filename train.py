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
from etaprogress.progress import ProgressBar

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

FLAGS = flags.FLAGS


def isBarracksSelected(obs):
    if obs["single_select"][0][0] == _TERRAN_BARRACKS:
        return True
    else:
        for single in obs["multi_select"]:
            if single[0] == _TERRAN_BARRACKS:
                return True
    return False


def getTerrainBarracksLocation(obs):
    if any(_TERRAN_BARRACKS in sublist for sublist in obs["screen"][_UNIT_TYPE_VIEW]):
        return zip(*np.where(obs["screen"][_UNIT_TYPE_VIEW] == _TERRAN_BARRACKS))


# Train the network
def train():
    FLAGS(sys.argv)
    # progress bar
    bar = ProgressBar(num_episodes)
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
                        visualize=visualize) as env:

        action_spec = env.action_spec()

        # Set the rate of random action decrease.
        e = startE
        stepDrop = (startE - endE) / annealing_steps

        # create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        total_steps = 0

        # Make a path for our model to be saved in.
        if not os.path.exists(path):
            os.makedirs(path)

        with tf.Session() as sess:
            sess.run(init)
            if load_model == True:
                print('Loading Model...')
                ckpt = tf.train.get_checkpoint_state(path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(num_episodes):
                bar.numerator = i
                print bar
                sys.stdout.flush()
                episodeBuffer = experience_buffer()
                # Reset environment and get first new observation
                timestep = env.reset()[0]
                s = processState(timestep.observation["screen"][_UNIT_TYPE_VIEW])
                d = False  # final step episode check
                rAll = 0
                j = 0
                # The Q-Network
                while j < max_epLength:  # TODO If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                    j += 1
                    avail_actions = timestep.observation["available_actions"]
                    # Choose an action by greedily (with e chance of random action) from the Q-network
                    if np.random.rand(1) < e or total_steps < pre_train_steps:
                        # Random actions
                        # if timestep.observation["single_select"]:
                        if _TERRAN_BARRACKS in s:
                            # The screen has the _TERRAN_BARRACKS
                            if not isBarracksSelected(timestep.observation):
                                # not selected yet.
                                location = getTerrainBarracksLocation(timestep.observation)
                                # y goes first, and then x
                                unit_y, unit_x = (
                                    timestep.observation["screen"][_UNIT_TYPE_VIEW] == _TERRAN_BARRACKS).nonzero()
                                target = [int(unit_x.mean()), int(unit_y.mean())]
                                a = _SELECT_POINT
                                args = [[_ALL_OF_TYPE_SELECT_POINT], target]
                            else:
                                # selected, good -> build marines
                                if _BUILD_MARINE in avail_actions:
                                    a = _BUILD_MARINE
                                    args = [[np.random.randint(0, size) for size in arg.sizes]
                                            for arg in action_spec.functions[a].args]
                                else:
                                    a = np.random.choice(avail_actions)
                                    args = [[np.random.randint(0, size) for size in arg.sizes]
                                            for arg in action_spec.functions[a].args]

                        else:
                            # No barracks avail
                            a = np.random.choice(avail_actions)
                            args = [[np.random.randint(0, size) for size in arg.sizes]
                                    for arg in action_spec.functions[a].args]

                    else:
                        # Predict
                        # TODO: what if prediction is not in scope?
                        a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
                        if not (a in avail_actions):
                            # random if not in actions
                            # print "prediction not valid, random action used"
                            a = np.random.choice(avail_actions)
                        args = [[np.random.randint(0, size) for size in arg.sizes]
                                for arg in action_spec.functions[a].args]
                    a_call = actions.FunctionCall(a, args)
                    timestep = env.step(actions=[a_call])[0]
                    r = timestep.reward
                    d = timestep.last()
                    s1 = processState(timestep.observation["screen"][_UNIT_TYPE_VIEW])
                    total_steps += 1
                    episodeBuffer.add(
                        np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.

                    if total_steps > pre_train_steps:
                        if e > endE:
                            e -= stepDrop

                        if total_steps % (update_freq) == 0:
                            trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                            # Below we perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                            Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                            end_multiplier = -(trainBatch[:, 4] - 1)
                            doubleQ = Q2[range(batch_size), Q1]
                            targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                            # Update the network with our target values.
                            _ = sess.run(mainQN.updateModel, \
                                         feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                                    mainQN.targetQ: targetQ,
                                                    mainQN.actions: trainBatch[:, 1]})

                            updateTarget(targetOps, sess)  # Update the target network toward the primary network.
                    rAll += r
                    s = s1

                    if d == True:
                        break

                myBuffer.add(episodeBuffer.buffer)
                jList.append(j)
                rList.append(rAll)
                # Periodically save the model.
                if i % 1000 == 0:
                    saver.save(sess, path + '/model-' + str(i) + '.ckpt')
                    np.save("rList.npy", rList)  # save r value
                    np.save("jList.npy", jList) #save j value
                    print("Saved Model")
                if len(rList) % 10 == 0:
                    print(total_steps, np.mean(rList[-10:]), e, pre_train_steps)
            saver.save(sess, path + '/model-' + str(i) + '.ckpt')
        print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")


if __name__ == "__main__":
    train()
