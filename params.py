batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
screen_xy = 84 #screen xy
unit_type_valid = [  0,  18,  19,  21,  45,  48, 341] #valid type array
unit_type_valid_size = len(unit_type_valid) #valid type array size
# unit_type_size = 1850 # unit type for each single pixel
state_size = screen_xy*screen_xy*unit_type_valid_size #Screen Size * Unit Type
# action_size = 524 #action size for sc2
action_type_valid = [0,
 1,
 2,
 3,
 4,
 5,
 6,
 7,
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