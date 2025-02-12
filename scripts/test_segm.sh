#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`

NUM_GPUS=1

python ./test.py --plugins='lapnet' --model_dir='/{path_to_files}/' --checkpoint_at=-1