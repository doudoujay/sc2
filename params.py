import pysc2.lib.features as features
import pysc2.lib.actions as actions
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
num_episodes = 10000 #How many episodes of game environment to train network with.
step_mul=8
max_epLength = 15000/step_mul #The max allowed length of our episode.
annealing_steps = max_epLength*100 #How many steps of training to reduce startE to endE.
pre_train_steps = max_epLength*100 #How many steps of random actions before training begins.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
screen_xy = 64 #screen xy
unit_type_valid = [  0,  18,  19,  21,  45,  48, 341] #valid type array
unit_type_valid_size = len(unit_type_valid) #valid type array size
# unit_type_size = 1850 # unit type for each single pixel
state_size = screen_xy*screen_xy #Screen Size * Unit Type
# action_size = 524 #action size for sc2
action_type_valid = [
 0, #no_op
 1,
 2,
 3,
 4,
 5,
 6, #select_idle_worker
 7, #
 10,
 11,
 12,
 13,
 42,
 91,
 264,
 269,
 274,
 331,
 332,
 333,
 334,
 343,
 344,
 451,
 452,
 453,
 477,
 490] #valid action type
action_type_valid_size = len(action_type_valid)  #valid action type size

map_name = "BuildMarines"
visualize = True
