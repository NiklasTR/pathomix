#!/bin/bash

mkdir -p ~/bucket/TCGA

export PATHOMIX_DATA=/home/ubuntu/bucket

aws configure

# download TCGA downloading tool
wget https://gdc.cancer.gov/system/files/authenticated%20user/0/gdc-client_v1.5.0_Ubuntu_x64.zip
unzip gdc-client_v1.5.0_Ubuntu_x64.zip

export GDC_PATH=/home/ubuntu/gdc-client

export MANIFEST_PATH=/home/ubuntu/pathomix/data/TCGA_manifests/manifest_TCGA_COAD.txt

python /home/ubuntu/pathomix/pathomix/preprocessing/utils/download_from_TCGA.py

