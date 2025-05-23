import datetime

from sigopt import Connection
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import backend as K
from keras import optimizers
from importlib.machinery import SourceFileLoader
import tensorflow.keras.callbacks as callb

import keras

import efficientnet.keras as efn

from models_own.eff_net import EffNetFT

cf = SourceFileLoader('cf', 'configs/sig_opt_ft_config.py').load_module()

def evaluate_model(assignments, steps_per_epoch_train, epochs, train_generator, validation_generator,
                   steps_per_epoch_val, tensor_board_callback, momentum, nesterov, model_path, workers=4,
                   use_multiprocessing=False, max_queue_size=100):
    model = EffNetFT(steps_per_epoch_train,
                 epochs,
                 train_generator,
                 validation_generator,
                 steps_per_epoch_val,
                 tensor_board_callback,
                 momentum=momentum,
                 nesterov=nesterov,
                 model_path=model_path,
                 workers=workers,
                 use_multiprocessing=use_multiprocessing,
                 max_queue_size=max_queue_size
                 )

    model_trained = model.train(assignments)
    acc = model_trained.evaluate_generator(generator=validation_generator, verbose=1, steps=len(validation_generator),
                                           max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)[1]
    return acc


efficient_net_type = cf.efficient_net_type
batch_size_ft = cf.batch_size_ft
train_folder = cf.train_folder
test_folder = cf.test_folder
random_seed = cf.random_seed
width_shift_range = cf.width_shift_range
input_size = cf.image_size

steps_per_epoch_train = cf.steps_per_epoch_train_ft
steps_per_epoch_val = cf.steps_per_epoch_val_ft
epochs = cf.epochs_ft
workers = cf.workers
use_multiprocessing = cf.use_multiprocessing
max_queue_size = cf.max_queue_size

model_path = cf.out_path_ft

momentum = cf.momentum
nesterov = cf.nesterov

'''
# determine input size from model name. input size is fixed
if efficient_net_type == 'B0':
    input_size = 224  # input size needed for network in pixels
    width_shift_range = 0
    height_shift_range = 0
    batch_size_ul = 512 # for p2
    #batch_size_ft = 64 # for p2
    batch_size_ft = batch_size_ft 
elif efficient_net_type == 'B3':
    input_size = 300
elif efficient_net_type == 'B4':
    input_size = 380
    batch_size_ul = 32
    batch_size_ft = 8
'''

# insert token here
conn = Connection(client_token="MBPBJXVLBQAJDOJNMRXQQXNCLQUOZFLYFMCZUWBJWKIVBKTC")

# general parameters for sigopt
observation_budget = cf.observation_budget
experiment_name = cf.experiment_name
project_name = cf.project_name
continue_experiment = cf.continue_experiment

# define range for decay parameters for hyperparameter search
lr_upper = cf.lr_upper
lr_lower = cf.lr_lower
lower_decay = cf.lower_decay
upper_decay = cf.upper_decay

if not continue_experiment:
    experiment = conn.experiments().create(
        name=experiment_name,
        # Define which parameters you would like to tune
        parameters=[
            dict(name='lr', type='double', bounds=dict(min=lr_lower, max=lr_upper)),
            dict(name='decay', type='double', bounds=dict(min=lower_decay, max=upper_decay)),
        ],
        metrics=[dict(name='accuracy', objective='maximize')],
        parallel_bandwidth=1,
        # Define an Observation Budget for your experiment
        observation_budget=observation_budget,
        project=project_name,
    )
    print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)

#create data generators

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
        batch_size=batch_size_ft,
        class_mode='binary',
        shuffle=True,
        seed=random_seed,
        )

validation_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=(input_size, input_size),
        batch_size=batch_size_ft,
        class_mode='binary',
        )

log_dir = "logs/fit/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S_{}_ft_sigopt".format(efficient_net_type))  # log dir for tensorboard
tensor_board_callback = callb.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

'''
#
# remove afterwards
#

assignments = {'lr': 0.001, 'decay': 0.00001}
value = evaluate_model(assignments, steps_per_epoch_train=steps_per_epoch_train, epochs=epochs, train_generator=train_generator,
                       validation_generator=validation_generator, steps_per_epoch_val=steps_per_epoch_val,
                       tensor_board_callback=tensor_board_callback, momentum=momentum, nesterov=nesterov,
                       model_path=model_path)

'''

for _ in range(observation_budget):
    if continue_experiment:
        experiment_id = "135957"
    else:
        experiment_id = experiment.id
    suggestion = conn.experiments(experiment_id).suggestions().create()
    assignments = suggestion.assignments
    print('current assignments: {}'.format(assignments))
    value = evaluate_model(assignments, steps_per_epoch_train=steps_per_epoch_train, epochs=epochs, train_generator=train_generator,
                           validation_generator=validation_generator, steps_per_epoch_val=steps_per_epoch_val,
                           tensor_board_callback=tensor_board_callback, momentum=momentum, nesterov=nesterov,
                           model_path=model_path, workers=workers, use_multiprocessing=use_multiprocessing,
                           max_queue_size=max_queue_size)

    conn.experiments(experiment_id).observations().create(
        suggestion=suggestion.id,
        value=value
    )

assignments = conn.experiments(experiment_id).best_assignments().fetch().data[0].assignments

print('best assignments')
print(assignments)
# This is a SigOpt-tuned model
#classifier = create_model(assignments)