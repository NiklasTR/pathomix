import os
import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
#import tensorflow.keras.callbacks as callb
import keras.callbacks as callb
import keras

import efficientnet.keras as efn

from train_ultimate_layers import train_ultimate_layers
from fine_tune_model import fine_tune_model

# general notes
# ratio mss/(mss+msi) = 0.58, accs above this value are better than guessing.
# not given that they are significantly better than guessing

# general parameters
efficient_net_type = 'B0'
image_size = 224    # actual image size in pixels
train_folder = '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/TRAIN_split'
test_folder = '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/VALIDATION'

# parameters for ultimate layer training
num_of_dense_layers = 0
dense_layer_dim = 32
epochs_ul = 1
steps_per_epoch_train_ul = 500
#steps_per_epoch_train_ul = 5
steps_per_epoch_val_ul = 20
out_path = './model_ultimate_with_proper_validation_{}'.format(efficient_net_type)

# parameters for fine tuning training
#epochs_ft = 40*8*4
epochs_ft = 4
steps_per_epoch_train_ft = 8
#steps_per_epoch_train_ft = 5
steps_per_epoch_val_ft = 8

out_path_ft = './model_fine_tuned_with_proper_validation_{}'.format(efficient_net_type)

# shifting for data augmentation, will be set to 0 in efficient_net_type == 'B0'
width_shift_range = 10
height_shift_range = 10
# determine input size from model name. input size is fixed
if efficient_net_type == 'B0':
    input_size = 224  # input size needed for network in pixels
    width_shift_range = 0
    height_shift_range = 0
    batch_size_ul = 512 # for p2
    batch_size_ft = 64 # for p2
elif efficient_net_type == 'B3':
    input_size = 300
elif efficient_net_type == 'B4':
    input_size = 380
    batch_size_ul = 32
    batch_size_ft = 8

lr = 10**(-3)
# interval size 0.4286, in paper: ration decay/lr ~ 10*(-6) to 10**(-3) at a batch size of 256
# the last term : batch_size /256 is only an approximation for the difference in batch size, since we do not have a linear decay
decay = 10**(-4.5) * lr *batch_size_ft/256.
momentum = 0.9
nesterov = True

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        width_shift_range=width_shift_range,
        height_shift_range=width_shift_range,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        cval=0
        )

test_datagen = ImageDataGenerator(rescale=1./255,
                                  fill_mode='constant',
                                  cval=0
                                  )

train_generator = train_datagen.flow_from_directory(
        train_folder,  # this is the target directory
        target_size=(input_size, input_size),  # all images will be resized to 150x150
        batch_size=batch_size_ul,
        class_mode='binary',
        shuffle=True,
        seed=42
        )

validation_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=(input_size, input_size),
        batch_size=batch_size_ul,
        class_mode='binary'
        )

model_in_cache = False
if not os.path.isfile('{}.json'.format(out_path)):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_{}_ul".format(efficient_net_type))  # log dir for tensorboard
    tensor_board_callback_ul = callb.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)

    # choose model from
    # https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
    if efficient_net_type == 'B0':
        model = efn.EfficientNetB0(weights='imagenet')
    elif efficient_net_type == 'B3':
        model = efn.EfficientNetB3(weights='imagenet')
    elif efficient_net_type == 'B4':
        model = efn.EfficientNetB4(weights='imagenet')

    model_ultimate = train_ultimate_layers(model=model,
                                           train_generator=train_generator,
                                           validation_generator=validation_generator,
                                           steps_per_epoch_train=steps_per_epoch_train_ul,
                                           steps_per_epoch_val=steps_per_epoch_val_ul,
                                           out_path=out_path,
                                           epochs=epochs_ul,
                                           num_of_dense_layers=num_of_dense_layers,
                                           dense_layer_dim=dense_layer_dim,
                                           tensor_board_callback=tensor_board_callback_ul,
                                           bsave=True)

    model_in_cache = True
'''
if not model_in_cache:
    # load json and create model
    json_file = open('{}.json'.format(out_path), 'r')
    loaded_model_json = json_file.read()
    json_file.close()


    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print('load ultimate model from {}'.format(out_path))
    loaded_model.load_weights("{}.h5".format(out_path))
    print("Loaded model from disk")

else:
    loaded_model = model_ultimate

train_generator = train_datagen.flow_from_directory(
        train_folder,  # this is the target directory
        target_size=(input_size, input_size),  # all images will be resized to 150x150
        batch_size=batch_size_ft,
        class_mode='binary',
        shuffle=True,
        seed=42,
        )

validation_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=(input_size, input_size),
        batch_size=batch_size_ft,
        class_mode='binary',
        )

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_{}_ft".format(efficient_net_type)) # log dir for tensorboard
tensor_board_callback_ft = callb.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)

fine_tuned_model = fine_tune_model(model=loaded_model,
                                   train_generator=train_generator,
                                   validation_generator=validation_generator,
                                   steps_per_epoch_train=steps_per_epoch_train_ft,
                                   steps_per_epoch_val=steps_per_epoch_val_ft,
                                   out_path=out_path_ft,
                                   lr=lr,  # in paper between 1e-10 - 0.1
                                   decay=decay,   # interval size 0.4286
                                   momentum=momentum,
                                   nesterov=nesterov,
                                   epochs=epochs_ft,
                                   tensor_board_callback=tensor_board_callback_ft,
                                   bsave=True)
'''



