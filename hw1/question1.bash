#!/bin/bash

source .env/bin/activate

environments=(Ant-v2 Walker2d-v2)

set -eux
for e in ${environments[*]}
do
    python run_exper.py experts/$e.pkl $e --num_rollouts=5
    python behavioral_cloning.py $e
    python run_clone.py $e
done

python create-table.py ${environments[*]}

for e in ${environments[*]}
do
    python hyper_parameter_search.py $e
done

python create-graph.py ${environments[*]}
