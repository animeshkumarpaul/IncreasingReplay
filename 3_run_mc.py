################################################################################################################
#                                                                                                              #
# Taken from  https://github.com/dongminlee94/deep_rl                                                          #
# Main modification is done by Animesh Kumar Paul (animeshk@ualberta.ca) - Denoted by AKP_ADDED/AKP_MODIFIED   #
# Thanks Videh Raj Nema (nema@ualberta.ca) for making contribution. His contribution is denoted by VRN_ADDED   #
# Added tag denotes the newly added codes.                                                                     #
################################################################################################################


import argparse
import torch
import json
import os
import gym
import time, datetime
import pandas as pd
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from common.utils import *
from dqn_ddqn import Agent

###########################################Configurations#####################################################################
## Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in MountainCar environment')

parser.add_argument('--exp', type=str, default=1,
                    help='experiment number based on replay hyperparameter')
parser.add_argument('--run_start', type=int, default=0,
                    help='run start')
parser.add_argument('--console_output', type=int, default=1,
                    help='Print log of training and evaluation (default: it will print)')
parser.add_argument('--is_mac', type=int, default=1,
                    help='Run on a mac or not (default: run on a mac)')
parser.add_argument('--use_gpu', type=int, default=0,
                    help='Run on a gpu or not (default: run on a cpu)')
parser.add_argument('--replay_frequency', type=int, default=1,
                    help='how many replay per step to train the agent')
parser.add_argument('--parent_path', type=str, default= r"./results", # if linux/mac, remove r before the path address
                    help='Path to parent directory where everything will be stored')

parser.add_argument('--save_json', type = bool, default = True,
    help='Save settings to file in json format. Ignored in json file')
parser.add_argument('--env', type=str, default= "MountainCar-v0",
                    help='MC environment')
parser.add_argument('--algo', type=str, default='dqn', 
                    help='select an algorithm among dqn, ddqn, a2c')
parser.add_argument('--phase', type=str, default='train',
                    help='choose between learning and offline evaluation phase')
parser.add_argument('--render', action='store_true', default=False,
                    help='if you want to render, set this to True')
parser.add_argument('--load', type=str, default=None,
                    help='copy & paste the path (str) to saved model and load it')
parser.add_argument('--gamma', type=float, default=0.99, 
                    help='gamma')
parser.add_argument('--learning_rate', type=float, default=1e-3, 
                    help='learning rate')
parser.add_argument('--gradient_momentum', type=float, default=0.9, 
                    help='gradient momentum')
parser.add_argument('--squared_gradient_momentum', type=float, default=0.999, 
                    help='squared gradient momentum')
parser.add_argument('--epsilon', type=float, default=1.0, 
                    help='initial value of epsilon')
parser.add_argument('--epsilon_decay', type=float, default=0.999, 
                    help='epsilon decay')
parser.add_argument('--final_epsilon', type=float, default=0.1, 
                    help='final value of epsilon')
parser.add_argument('--replay_capacity', type=int, default=4000, 
                    help='replay_capacity')
parser.add_argument('--batch_size', type=int, default=32, 
                    help='batch_size')
parser.add_argument('--target_update_step', type=int, default=128, 
                    help='target_update_step')
parser.add_argument('--save_model_per_step', type=int, default=10000, 
                    help='save_model_per_step')
parser.add_argument('--max_runs', type=int, default=30, 
                    help='how many agents are trained')
parser.add_argument('--max_interactions', type=int, default=250000,
                    help='max_interaction to run and train agent')
parser.add_argument('--replay_start_size', type=int, default=1024,
                    help='replay_start_size')
parser.add_argument('--eval_per_train', type=int, default=20000, 
                    help='every these number of steps, do offline evaluation')
parser.add_argument('--evaluation_episodes', type=int, default=20, 
                    help='number of episodes for evaluation')
parser.add_argument('--max_step', type=int, default=2000,
                    help='max episode step')
parser.add_argument('--l1_size', type=int, default=32,
                    help='first hidden layer size')
parser.add_argument('--l2_size', type=int, default=32,
                    help='second hidden layer size')
parser.add_argument('--threshold_return', type=int, default=-120, 
                    help='solved requirement for success in given environment')
parser.add_argument('--tensorboard', action='store_true', default=True)
parser.add_argument('--gpu_index', type=int, default=0)
args = parser.parse_args()
################################################################################################################



#VRN_ADDED: To run on MAC mps.
if args.use_gpu:
    if args.is_mac: #AKP_ADDED
        if torch.has_mps:
            # for Apple metal in Macbook
            device = torch.device('mps')
        else:
            # for CPU
            device = torch.device('cpu')
        # for Nvidia GPU
    elif torch.cuda.is_available():
        device = torch.device('cuda', index=args.gpu_index)
    else:
        # for CPU
        device = torch.device('cpu')
else: #AKP_ADDED
    device = torch.device('cpu')



#device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
print(device)


# Specifying paths for seeds and experiments
seed_path = args.parent_path + '/Seeds'
exp_path = args.parent_path + '/exp{}_{}'.format(args.exp, args.algo) # + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#exp_path = args.parent_path + '/exp{}_{}_'.format(args.exp, args.algo) # + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
if not os.path.exists(exp_path):
    os.makedirs(exp_path)


"""
How does it work?:
seeds_array_runs: This has one seed for each run. 
It is used in the seed_everything(seed_array_runs[run]) function to set seed for the everything at the beginning of run.
This will fix the seed for agent (NN) at the start of the run, torch seed, numpy seed, torch cuda seed etc.

# MODIFY
seeds_array_training: This has one seed for each training episode. 
This spreads across runs, i.e. if the first run has 20 episodes, they will use the first 20 seeds from this seed file. 
The first episode of 2nd run will start with the 21st seed in the seed file.

seeds_array_eval: This has one seed for each evaluation episode. 
These seeds stay the same across runs, i.e. if there are 20 evaluation episodes, every time the evaluation function is called 
(does not matter what the current run index is), the episode is run using these 20 seeds. 
"""
#AKP_ADDED: Loading the seeds
seeds_array_training = load_path_obj(seed_path + '/seeds_array_training.pkl')
seeds_array_eval = load_path_obj(seed_path + '/seeds_array_eval_1000.pkl')
seeds_array_runs = load_path_obj(seed_path + '/seeds_array_runs.pkl')


#VRN_ADDED: Create json file
def log_hps(path):
    filename = path + '/Hyperparameters.json'
    json_dict = {}
    # log HPs
    json_dict['Hyperparameters'] = {}
    for arg in vars(args):
        json_dict['Hyperparameters'][str(arg)] = str(getattr(args, arg))
    # log device details
    json_dict['Device details'] = {}
    if args.is_mac:
        if device == torch.device('mps'):
            json_dict['Device details']['Device'] = 'mps'
        elif device == torch.device('cpu'):
            json_dict['Device details']['Device'] = 'cpu'
    elif device == torch.device('cpu'):
            json_dict['Device details']['Device'] = 'cpu'
    else:
        json_dict['Device details']['Device'] = {}
        current_device = torch.cuda.current_device()
        json_dict['Device details']['Device']['current device'] = str(torch.cuda.device(current_device))
        json_dict['Device details']['Device']['device count'] = str(torch.cuda.device_count())
        json_dict['Device details']['Device']['device name'] = str(torch.cuda.get_device_name(current_device))
    
    # final dump
    with open(filename, 'w') as ptr:
        json.dump(json_dict, ptr, indent=4)

################################################################################################################


def run_experiment():
    """Run the experiment"""
    #AKP_ADDED: For multiple Runs
    run = args.run_start -1 
    

    #AKP_ADDED:  Dataframes for logging
    log_train = pd.DataFrame(columns=['Episode', 'Metric', 'Run', 'Seed_idx', 'Step', 'Undiscount_Return'])
    
    #VRN_ADDED: log HPs
    hp_path = exp_path
    if args.save_json:
        log_hps(hp_path)
        
    #AKP_ADDED: Run the loop maximum number of max_runs.
    #args.max_runs = run + 1
    while run < args.max_runs:
        seed_training_index = run * 4000000 #AKP_ADDED: Have a fixed seed values for same run number
        run = run + 1

        #VRN_ADDED: allocate paths for logging
        run_path = exp_path + '/Run_{}'.format(run)
        nn_params_path = run_path + '/NN_params'
        data_path = run_path + '/Data'
        tensorboard_path = run_path + '/Tensorboard'
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            os.makedirs(nn_params_path)
            os.makedirs(data_path)
            os.makedirs(tensorboard_path)
        paths = [nn_params_path, data_path, tensorboard_path]

        print('---------------------------------------')
        print('Run:', str(run))
        print('---------------------------------------')

        # Initialize environment
        #AKP_MODIFY: Use .env to avoid termination after 200 episodes
        env = gym.make(args.env).env 
        obs_dim = env.observation_space.shape[0]
        act_num = env.action_space.n

        print('---------------------------------------')
        print('Environment:', args.env)
        print('Algorithm:', args.algo)
        print('State dimension:', obs_dim)
        print('Action number:', act_num)
        print('---------------------------------------')

        #AKP_ADDED Set a random seed for the run
        seed_everything(seeds_array_runs[run], env) 

        # Create an agent
        agent = Agent(env, args, device, obs_dim, act_num, paths, run, seeds_array_eval)

        # If we have a saved model, load it
        if args.load is not None:
            pretrained_model_path = str(args.load)
            pretrained_model = torch.load(pretrained_model_path, map_location=device)
            if args.algo == 'dqn' or args.algo == 'ddqn':
                agent.qf.load_state_dict(pretrained_model)
            else:
                agent.policy.load_state_dict(pretrained_model)

        # Create a SummaryWriter object by TensorBoard
        if args.tensorboard and args.load is None:
            dir_path = tensorboard_path + '/' + args.env \
                               + args.algo \
                               + str(run) \
                               + '_run_seed_' + str(seeds_array_runs[run]) \
                               + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            writer = SummaryWriter(log_dir=dir_path)

        start_time = time.time()

        #AKP_ADDED
        train_num_steps = 0
        train_sum_returns = 0
        train_num_episodes = 0

        # Main loop
        #AKP_MODIFY: Change the base code from episodes to interactions.
        # It will terminate after a fixed number of total interactions instead of episodes to have a fair comparison between different learning agent.
        while 1:
            if agent.steps >= args.max_interactions:
                break

            # Perform the training phase, during which the agent learns
            if args.phase == 'train':
                agent.eval_mode = False

                # Run one episode
                #AKP_ADDED: Get the seed index for this episode from the previously generated seeds file.
                training_eps_seed = seeds_array_training[seed_training_index]

                train_step_length, train_episode_return, log_state_visitation = agent.run_episode(args.max_step, args.max_interactions, writer, training_eps_seed=training_eps_seed) ###Change: 

                train_num_steps += train_step_length
                train_sum_returns += train_episode_return
                train_num_episodes += 1
                train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

                # Log experiment result for training episodes
                if args.tensorboard and args.load is None:
                    writer.add_scalar('Train/AverageReturns', train_average_return, train_num_steps) #AKP_MODIFY: change from episode to number of step 
                    writer.add_scalar('Train/EpisodeReturns', train_episode_return, train_num_steps) #AKP_MODIFY: change from episode to number of step
                    
                    #AKP_ADDED
                    df = pd.DataFrame({'Episode': train_num_episodes, 'Metric': 'EpisodeReturns', 'Run': run, 'Seed_idx': seed_training_index, 'Step': train_num_steps, 'Undiscount_Return': train_episode_return}, index=[0])
                    log_train = pd.concat([log_train.loc[:], df]).reset_index(drop=True)
                    df = pd.DataFrame({'Episode': train_num_episodes, 'Metric': 'AverageReturns', 'Run': run, 'Seed_idx': seed_training_index, 'Step': train_num_steps, 'Undiscount_Return': train_average_return}, index=[0])
                    log_train = pd.concat([log_train.loc[:], df]).reset_index(drop=True)
                    log_train.to_pickle(data_path + '/log_train.pkl')
                    #log_state_visitation_train.to_pickle(data_path + '/log_state_visitation_train.pkl')

                    #VRN_ADDED
                    np.save(data_path + '/log_state_visitation_train.npy', np.array(log_state_visitation['train'])) # use np.load(filename, allow_pickle=True) to extract, then use .item() to extract
                    seed_training_index += 1
                    
                #eval(agent, args, writer)
                
                if args.phase == 'train' and args.console_output == 1:
                    print('---------------------------------------')
                    print('Interactions:', train_num_steps)
                    print('Episodes:', train_num_episodes)
                    print('EpisodeReturn:', round(train_episode_return, 2))
                    print('AverageReturn:', round(train_average_return, 2))
                    print('OtherLogs:', agent.logger)
                    print('Time:', int(time.time() - start_time))
                    print('---------------------------------------')              
        env.close()

        #AKP_ADDED & VRN_MODIFY
        # save final NN parameters
        file_path = nn_params_path + '/Final_' + 'Q_Network'+ '_exp_' + str(args.exp) + '_run_'+ str(run) + '_replay_frequency_'+ str(args.replay_frequency) + '_steps_' + str(agent.steps) + '.pt'
        save_model(agent.qf, file_path)
        file_path = nn_params_path + '/Final_' + 'Target_Q_Network'+ '_exp_' + str(args.exp) + '_run_'+ str(run) + '_replay_frequency_'+ str(args.replay_frequency) + '_steps_' + str(agent.steps) + '.pt'
        save_model(agent.qf_target, file_path)

run_experiment()
