#!/bin/bash

mkdir -p ~/bucket/WSI_preparation

export PATHOMIX_DATA=/home/ubuntu/bucket
export FONT_PATH=/home/ubuntu/pathomix/pathomix/preprocessing/docs/arial.ttf

mkdir -p ~/virtual_envs

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales

sudo apt-get install python3-venv

python3 -m venv ~/virtual_envs/venv_pathomix

source ~/virtual_envs/venv_pathomix/bin/activate
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