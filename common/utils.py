################################################################################################################
#                                                                                                              #
# Taken from  https://github.com/dongminlee94/deep_rl                                                          #
# Modified By Animesh Kumar Paul (animeshk@ualberta.ca)                                                        #
# Added tag denotes the newly added codes
################################################################################################################


import torch
import torch.nn as nn
import pickle
import random
import numpy as np
import os
import copy

def hard_target_update(main, target):
    target.load_state_dict(main.state_dict())

def soft_target_update(main, target, tau=0.005):
    for main_param, target_param in zip(main.parameters(), target.parameters()):
        target_param.data.copy_(tau*main_param.data + (1.0-tau)*target_param.data)

def load_path_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#AKP_ADDED: Taken from stackoverflow
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

#AKP_ADDED
def save_model(model, file_path):
    torch.save(copy.deepcopy(model).cpu(), file_path)

#AKP_ADDED: Taken from https://stackoverflow.com/questions/47331235/how-should-openai-environments-gyms-use-env-seed0
def seed_everything(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)
