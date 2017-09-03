from pysc2.env import sc2_env
from loop import loop
from pysc2.agents import random_agent
from gflags import exceptions

steps = 2000
step_mul = 16


def test_build_marine_random():

    with sc2_env.SC2Env(
            "BuildMarines",
            step_mul=step_mul,
            game_steps_per_episode=steps * step_mul) as env:

        env.observation_spec()

        agent = random_agent.RandomAgent()
        loop([agent], env, steps)
        print agent.reward

if __name__ == "__main__":
    test_build_marine_random()