# Understanding the effect of varying amounts of replay per step in DQN

I acknowledge that this work is part of the credited course group project on Reinforcement learning II at the University of Alberta. 

Group Memeber:
a) Animesh Kumar Paul <animeshk@ualberta.ca>
b) Videh Raj Nema <nema@ualberta.ca>
 

## Usage
Run codes one of the following ways:

### Option 1: Execute all runs at once
python 3_run_mc_colab.py --exp experiment_name --algo dqn  --replay_frequency 2 --learning_rate 0.1 --console_output 0 --use_gpu 0 --is_mac 0 --run_start 1


### Option 2: Execute each run seperately manually
start = 1

max_runs = start + 1

python 3_run_mc_colab.py --exp experiment_name --algo dqn  --replay_frequency 2 --learning_rate 0.1 --console_output 0 --use_gpu 0 --is_mac 0 --max_runs max_runs --run_start start


### Option 3: Execute each run seperately automatically (Sequentially execute each run) -You need to set the hyper-paramters in 2_easy_run_sequentially.py file.
python 2_easy_run_sequentially.py

### Option 4: Execute each run seperately automatically (Parallelly execute each run)-You need to set the hyper-paramters in 2_easy_run_parallel_backgroun.py file.
python 2_easy_run_parallel_background.sh


It contains also two jupyter notebooks for plotting purposes.