#!/bin/bash

PATH=~/.mujoco/:$PATH python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --render --num_rollouts=50
