# downloading pre-tesselated and QCed data from https://zenodo.org/record/1214456#.Xbn63EVKg5k
wget -v https://zenodo.org/record/2530835/files/CRC_DX_TEST_MSIMUT.zip?download=1
wget -v https://zenodo.org/record/2530835/files/CRC_DX_TEST_MSS.zip?download=1
wget -v https://zenodo.org/record/2530835/files/CRC_DX_TRAIN_MSIMUT.zip?download=1
wget -v https://zenodo.org/record/2530835/files/CRC_DX_TRAIN_MSS.zip?download=1
wget -v https://zenodo.org/record/2530835/files/STAD_TEST_MSIMUT.zip?download=1
wget -v https://zenodo.org/record/2530835/files/STAD_TEST_MSS.zip?download=1
wget -v https://zenodo.org/record/2530835/files/STAD_TRAIN_MSIMUT.zip?download=1
wget -v https://zenodo.org/record/2530835/files/STAD_TRAIN_MSS.zip?download=1

# creating dirs
mkdir STAD
mkdir CRC_DX
mkdir STAD/TEST STAD/TRAIN
mkdir CRC_DX/TEST CRC_DX/TRAIN

# moving files
rsync --remove-source-files -rav STAD_TEST* STAD/TEST
rsync --remove-source-files -rav STAD_TRAIN* STAD/TRAIN
rsync --remove-source-files -rav CRC_DX_TRAIN* CRC_DX/TRAIN
rsync --remove-source-files -rav CRC_DX_TEST* CRC_DX/TEST

# unzipping
unzip 'STAD/TEST/*.zip*' -d STAD/TEST
unzip 'STAD/TRAIN/*.zip*' -d STAD/TRAIN
unzip 'CRC_DX/TEST/*.zip*' -d CRC_DX/TEST
unzip 'CRC_DX/TRAIN/*.zip*' -d CRC_DX/TRAIN

# cleaning zip
find . -type f -name '*.zip*' -delete

# the data is now distributed and can be used for training using common data generators
