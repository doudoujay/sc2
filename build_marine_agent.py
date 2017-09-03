from pysc2.agents import base_agent
from pysc2.lib import features
from pysc2.lib import actions
import numpy as np

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class build_marine_agent(base_agent.BaseAgent):
    def step(self, obs): #TODO
        super(build_marine_agent, self).step(obs)
        avail_actions = obs.observation["available_actions"]
        print avail_actions
        function_id = np.random.choice(avail_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)
