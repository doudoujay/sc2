"""A run loop for agent/environment interaction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time

rList = []

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
          return
        if timesteps[0].last(): #end of a episode
          # print(timestep.observation["screen"][6])
          rList.append(agent.reward)
          break
        timesteps = env.step(actions)
  except KeyboardInterrupt:
    pass
  finally:
    print(rList)
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))

