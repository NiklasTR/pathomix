# parameters needed for wandb sweep
debug = False
experiment = "tumor_detection"

project_name = "fine-tuning_hyperparameter"
model_out_name = None
optimizing_parameters = dict(
    lr=0.000006258,
    decay=2.630e-10,
    momentum=0.2681
)

# paramerts for data generator
data_gen_dict = dict(
    featurewise_center=False,
    samplewise_center=True,
    featurewise_std_normalization=False,
    samplewise_std_normalization=True,
    rotation_range_train=0,
    width_shift_range_train=0,
    height_shift_range_train=0,
    horizontal_flip_train=True,
    vertical_flip_train=True,
    fill_mode_train='constant',
    cval_train=0,
    rotation_range_val=0,
    width_shift_range_val=0,
    height_shift_range_val=0,
    horizontal_flip_val=False,
    vertical_flip_val=False,
    fill_mode_val='constant',
    cval_val=0,
    class_mode='categorical',
    x_col='relative_path',
    y_col='label',
    do_augmentation=True,
    devide_by=5,
)

# determined hyperparameters
if debug:
    seed=42
    batch_size_ul=8
    batch_size_ft=2
    input_size=(224, 224)
else:
    seed=42
    batch_size_ul=64
    batch_size_ft=4
    input_size=(456, 456)

hp_dict = dict(
    seed=seed,
    input_size=input_size,
    batch_size_ul=batch_size_ul,
    batch_size_ft=batch_size_ft,
    epochs_ul=3,
    epochs_ft=10,
    nesterov=False,
    verbose=2,
    validation_freq=1,
    class_weight=None,
    use_multiprocessing=False,
    shuffle=True,
    initial_epoch=0,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
    optimizer='rmsprop'
)

