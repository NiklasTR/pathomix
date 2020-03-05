#!/bin/bash

mkdir -p ~/bucket/Jakob_cancer_detection

export PATHOMIX_DATA=/home/ubuntu/bucket
export FONT_PATH=/home/ubuntu/pathomix/pathomix/preprocessing/docs/arial.ttf

#mkdir -p ~/virtual_envs

#python3 -m venv ~/virtual_envs/venv_pathomix

ve() { source activate $1; }
ve tensorflow2_p36

#source /home/ubuntu/virtual_envs/venv_pathomix/bin/activate
pip install --upgrade pip
#pip install -r requirements.txt

pip install wandb
pip install efficientnet
sudo apt-get install openslide-tools	
pip install Pillow
pip install openslide-python
pip install --upgrade scikit-image

wandb login

aws configure
aws s3 sync s3://evotec/pathomix/cancer_detection_jakob /home/ubuntu/bucket/Jakob_cancer_detection/


