from sigopt import Connection
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import optimizers

def create_model(assignments):
    lr = assignments['lr']
    decay = assignments['decay']
    # hard coded for now
    momentum = 0.9
    nesterov = True
    out_path = TODO
    json_file = open('{}.json'.format(out_path), 'r')
    loaded_model_json = json_file.read()
    json_file.close()


    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print('load ultimate model from {}'.format(out_path))
    loaded_model.load_weights("{}.h5".format(out_path))
    print("Loaded model from disk")


    sgd = optimizers.SGD(lr=lr, momentum=momentum, nesterov=nesterov, decay=decay)
    loaded_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    loaded_model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch_train,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=steps_per_epoch_val,
        callbacks=[tensor_board_callback])

def evalulate_model(assignments):
    model = create_model(assignments)
    acc = model.evaluate(x=validation_generator, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=4, use_multiprocessing=False)
    return acc


# general parameters
efficient_net_type = 'B0'
image_size = 224    # actual image size in pixels
train_folder = '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/TRAIN_split'
test_folder = '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/VALIDATION'

# parameters for ultimate layer training
num_of_dense_layers = 0
dense_layer_dim = 32
epochs_ul = 1
#steps_per_epoch_train_ul = 500
steps_per_epoch_train_ul = 5
steps_per_epoch_val_ul = 20
out_path = './model_ultimate_with_proper_validation_{}'.format(efficient_net_type)

# parameters for fine tuning training
#epochs_ft = 40*8*4
epochs_ft = 4
#steps_per_epoch_train_ft = 500
steps_per_epoch_train_ft = 5
steps_per_epoch_val_ft = 80

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

# insert token here
conn = Connection(client_token="MBPBJXVLBQAJDOJNMRXQQXNCLQUOZFLYFMCZUWBJWKIVBKTC")

lower_decay = 0.000001 * lr * batch_size_ft/256.
upper_decay = 0.001 * lr * batch_size_ft/256.

experiment = conn.experiments().create(
    name='Franke Optimization (Python)',
    # Define which parameters you would like to tune
    parameters=[
        dict(name='lr', type='double', bounds=dict(min=0.0001, max=0.1)),
        dict(name='decay', type='double', bounds=dict(min=lower_decay, max=upper_decay)),
    ],
    metrics=[dict(name='function_value', objective='maximize')],
    parallel_bandwidth=1,
    # Define an Observation Budget for your experiment
    observation_budget=30,
    project="MSI_GI_FFPE",
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
        seed=42,
        )

validation_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=(input_size, input_size),
        batch_size=batch_size_ft,
        class_mode='binary',
        )


for _ in range(experiment.observation_budget):
    suggestion = conn.experiments(experiment.id).suggestions().create()
    assignments = suggestion.assignments
    value = evaluate_model(assignments)

    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        value=value
    )

assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments

print(assignments)

# This is a SigOpt-tuned model
classifier = create_model(assignments)