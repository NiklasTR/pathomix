#!/bin/bash

mkdir -p ~/bucket/MSI_classification

export PATHOMIX_DATA=/home/ubuntu/bucket

mkdir -p ~/virtual_envs

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
#aws s3 sync s3://evotec/pathomix/cancer_detection_jakob /home/ubuntu/bucket/Jakob_cancer_detection/


# make train folder
cd ~/bucket/MSI_classification
mkdir train
cd train
wget -v https://zenodo.org/record/2530835/files/CRC_DX_TRAIN_MSIMUT.zip
unzip CRC_DX_TRAIN_MSIMUT.zip
rm CRC_DX_TRAIN_MSIMUT.zip
wget -v https://zenodo.org/record/2530835/files/CRC_DX_TRAIN_MSS.zip
unzip CRC_DX_TRAIN_MSS.zip   
rm CRC_DX_TRAIN_MSS.zip  

# make test folder
cd ..
mkdir test
cd test
wget -v https://zenodo.org/record/2530835/files/CRC_DX_TEST_MSIMUT.zip
unzip CRC_DX_TEST_MSIMUT.zip  
rm RC_DX_TEST_MSIMUT.zip 
wget -v https://zenodo.org/record/2530835/files/CRC_DX_TEST_MSS.zip
unzip CRC_DX_TEST_MSS.zip
rm CRC_DX_TEST_MSS.zip


