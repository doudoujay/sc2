from Qnetwork import *
from util_func import *
from experience_buffer import *
from pysc2.env import sc2_env
from pysc2 import maps
from pysc2.lib import actions
import os

# Train the network
def train():
    maps.get(map_name)  # Assert the map exists.

    tf.reset_default_graph()
    mainQN = Qnetwork(h_size)
    targetQN = Qnetwork(h_size)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables, tau)

    myBuffer = experience_buffer()

    env = sc2_env(map_name,
            agent_race=None,
            bot_race=None,
            difficulty=None,
            step_mul=8,
            game_steps_per_episode=0, #maxEpLength?
            screen_size_px=(screen_xy, screen_xy),
            minimap_size_px=(64, 64),
            visualize=visualize)
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
            episodeBuffer = experience_buffer()
            # Reset environment and get first new observation
            timestep = env.reset()
            s = processState(timestep.observation["screen"][6])
            d = timestep.last() #final step episode check
            rAll = 0
            j = 0
            # The Q-Network
            while j < max_epLength:  #TODO If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    avail_actions = timestep.observation["available_actions"]
                    function_id = np.random.choice(avail_actions)
                    #TODO Do we need to consider different args for every action as the variability for the actions?
                    args = [[np.random.randint(0, size) for size in arg.sizes]
                            for arg in action_spec.functions[function_id].args]
                    a = actions.FunctionCall(function_id, args)
                else:
                    a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]
                timestep = env.step(a)
                r = timestep.reward
                d = timestep.last()
                s1 = processState(timestep.observation["screen"][6])
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
                                     feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
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
                print("Saved Model")
            if len(rList) % 10 == 0:
                print(total_steps, np.mean(rList[-10:]), e)
        saver.save(sess, path + '/model-' + str(i) + '.ckpt')
    print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")