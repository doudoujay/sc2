"""A run loop for agent/environment interaction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import pysc2.lib.features as features

import time

rList = []
screenList = []
actionList = []
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index



def loop(agents, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  total_frames = 0
  start_time = time.time()

  action_spec = env.action_spec()
  observation_spec = env.observation_spec()
  for agent in agents:
    agent.setup(observation_spec, action_spec)

  try:
    while True:
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        total_frames += 1
        actions = [agent.step(timestep)
                   for agent, timestep in zip(agents, timesteps)]
        if max_frames and total_frames >= max_frames: #end of the total frames
          actionList.append(env._seen) #save action_seen for each agent
          return
        if timesteps[0].last(): #end of a episode
          screenList.append(timestep.observation["screen"][_PLAYER_RELATIVE])
          rList.append(timestep.reward)
          break
        timesteps = env.step(actions)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))
    numpy.save("rList.npy", rList) #save r value
    numpy.save("screenList.npy", screenList) # save screen value
    numpy.save("actionList.npy", actionList) # save action list

