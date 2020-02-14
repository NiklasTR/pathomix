#!/bin/bash

mkdir -p ~/bucket/Jakob_cancer_detection

export PATHOMIX_DATA=/home/ubuntu/bucket

mkdir -p ~/virtual_envs

python3 -m venv ~/virtual_envs/venv_pathomix

ve() { source $1/bin/activate; }
ve /home/ubuntu/virtual_envs/venv_pathomix

#source /home/ubuntu/virtual_envs/venv_pathomix/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

wandb login

aws configure
aws s3 sync s3://evotec/pathomix/cancer_detection_jakob /home/ubuntu/bucket/Jakob_cancer_detection/


