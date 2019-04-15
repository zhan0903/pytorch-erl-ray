#!/bin/bash
for ((i=0;i<1;i+=1))
do
        python main.py \
        --env_name "HalfCheetah-v2" \
        --seed $i \
        --output $i

        python main.py \
        --env_name "Hopper-v2" \
        --seed $i \
        --output $i

        python main.py \
        --env_name "Walker2d-v2" \
        --seed $i \
        --output $i

        python main.py \
        --env_name "Ant-v2" \
        --seed $i \
        --output $i

        python main.py \
        --env_name "Swimmer-v2" \
        --seed $i \
        --output $i
done