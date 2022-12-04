
################################################################################################################
#                                                                                                              #
# Author: Animesh Kumar Paul (animeshk@ualberta.ca)                                                            #
# Execute Each run seperately to make system faster                                                            #    
#                                                                                                              #
################################################################################################################
  
import os
import subprocess
import shlex, subprocess
import time
program_list=[]
start = 25
end = 25



print(start)
print(end)
for run in range(start,end+1):
   #If want to add additional arguments, you can add that here.
   max_runs = run + 1
   #program_list.append("python 3_run_mc.py --exp 1_lr --algo dqn  --replay_frequency 1 --max_interactions 5000 --max_step 200 --eval_per_train 1000 --save_model_per_step 1000 --console_output 0 --use_gpu 1 --is_mac 0 --run_start "+ str(run))
   #program_list.append("python 3_run_mc_colab.py --exp 1_lr_0.1 --algo dqn  --replay_frequency 1 --learning_rate 0.1 --console_output 0 --use_gpu 1 --is_mac 0 --run_start "+ str(run))
   program_list.append("python 3_run_mc_colab.py --exp 3_lr_0.001 --algo ddqn  --replay_frequency 4 --learning_rate 0.001 --console_output 0 --use_gpu 0 --is_mac 0 --max_runs "+ str(max_runs)+" --run_start "+ str(run))


for program in program_list:
    print("\nStart: " + program)
    subprocess.call(shlex.split(program))
    time.sleep(30)
    print("\nFinished: " + program)
