
################################################################################################################
#                                                                                                              #
# Author: Animesh Kumar Paul (animeshk@ualberta.ca)                                                            #
################################################################################################################

#Generate Pseudo Random Number Generator
import random
from datetime import datetime
import pickle
import os

run_path = './results/Seeds/'

#Save pickle file
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if not os.path.exists(run_path):
    os.makedirs(run_path)




# Passing the current time as the seed value
random.seed(datetime.now())
seeds_array=[]
#Generate Seeds for environement interactions inside the run
for i in range(200000000): #choice of the total number of seeds depends on applictions and developers
    seeds_array.append(random.randint(0, 200000000))
save_obj(seeds_array, run_path + 'seeds_array_training')

# Passing the current time as the seed value
random.seed(datetime.now())
seeds_array=[]
#Generate Seeds for environement interactions inside the run
for i in range(200000000): #choice of the total number of seeds depends on applictions and developers
    seeds_array.append(random.randint(0, 200000000))
save_obj(seeds_array, run_path + 'seeds_array_eval')

# Passing the current time as the seed value
random.seed(datetime.now())
seeds_array=[]
#Generate seeds for each run
for i in range(200): #choice of the total number of seeds depends on applictions and developers
    seeds_array.append(random.randint(0, 200))
save_obj(seeds_array, run_path + 'seeds_array_runs')