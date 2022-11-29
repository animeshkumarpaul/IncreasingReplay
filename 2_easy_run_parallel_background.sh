#!/bin/bash

for i in {1..30}; 
do 
     echo "Run $i" 
     nohup python 3_run_mc.py --exp 2_lr_0.0001 --algo dqn  --replay_frequency 2 --learning_rate 0.0001 --console_output 0 --use_gpu 0 --is_mac 0 --max_runs $(( $i + 1 )) --run_start "$i" > output_"i" &
     
done
