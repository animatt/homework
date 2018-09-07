#!/bin/bash

source .env/bin/activate

./demo.bash

environments=(Ant-v2 Walker2d-v2)

set -eux
for e in environments
do
    python behavioral_cloning.py $e
    python run-clone.py $e
done

python create-table.py ${environments[*]}
