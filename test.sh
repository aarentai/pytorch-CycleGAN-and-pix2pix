#! /bin/bash
PYTHON_INTERPRETER_PATH="/home/ubuntu/.conda/envs/p2p/bin/python"
PROJECT_ABSOLUTE_PATH="/home/ubuntu/hcdai/Projects/pytorch-CycleGAN-and-pix2pix"

$PYTHON_INTERPRETER_PATH \
    $PROJECT_ABSOLUTE_PATH/test.py \
    --dataroot $PROJECT_ABSOLUTE_PATH/datasets/$1 \
    --name $1"_pix2pix"\
    --model pix2pix \
    --direction AtoB 