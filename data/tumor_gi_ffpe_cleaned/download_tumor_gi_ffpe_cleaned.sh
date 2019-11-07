# downloading pre-tesselated and QCed data from https://zenodo.org/record/2530789#.XcNVHJJKg5k
wget -v https://zenodo.org/record/2530789/files/ADIMUC.zip?download=1
wget -v https://zenodo.org/record/2530789/files/STRMUS.zip?download=1
wget -v https://zenodo.org/record/2530789/files/TUMSTU.zip?download=1


# creating dirs
mkdir ADIMUC
mkdir STRMUS
mkdir TUMSTU
mkdir TRAIN
mkdir TEST
mkdir TRAIN/ADIMUC
mkdir TRAIN/STRMUS
mkdir TRAIN/TUMSTU
mkdir TEST/ADIMUC
mkdir TEST/STRMUS
mkdir TEST/TUMSTU

# unzipping
unzip TUMSTU.zip?download=1
unzip ADIMUC.zip?download=1
unzip STRMUS.zip?download=1

#moving files
shuf -zen3200 TUMSTU/* | xargs -0 mv -t TRAIN/TUMSTU
shuf -zen3200 STRMUS/* | xargs -0 mv -t TRAIN/STRMUS
shuf -zen3200 ADIMUC/* | xargs -0 mv -t TRAIN/ADIMUC
mv ADIMUC/* TEST/ADIMUC
mv STRMUS/* TEST/STRMUS
mv TUMSTU/* TEST/TUMSTU

#cleaning up
rm -r TUMSTU
rm -r STRMUS
rm -r ADIMUC

# cleaning zip
find . -type f -name '*.zip*' -delete

# the data is now distributed and can be used for training using common data generators
