import os
import numpy as np
import multiprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import efficientnet.keras as efn

import wandb
from wandb.keras import WandbCallback

from utils.experiment_setup import create_data_frame, split_data_frame, random_crop, DataLoader

def train_eff_net(experiment, cf, debug):
    '''
        parser = argparse.ArgumentParser(description='Give parameters for tumor detection fine tuning')
        parser.add_argument("--learning-rate", help="")
        parser.add_argument("--decay", help="")
        parser.add_argument("--momentum", help="")

        args = parser.parse_args()
        lr = args.lr
        decay = args.decay
        momentum = args.momentum
        '''
    # load parameters from config file
    data_gen_dict = cf.data_gen_dict
    hp_dict = cf.hp_dict
    optimizing_parameters = cf.optimizing_parameters

    if experiment == "tumor_detection":
        base_dir = os.path.join(os.environ['PATHOMIX_DATA'], 'Jakob_cancer_detection')
        data_dir_train = os.path.join(base_dir, 'train')
        data_dir_val = data_dir_train

        # prepare data for data loader
        df_total = create_data_frame(base_dir=data_dir_train)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=hp_dict["seed"])
        # get indices for train and validation
        train_idx, val_idx = next(kf.split(X=np.zeros(len(df_total)), y=df_total['label']))
        df_train, df_val = split_data_frame(df_total, train_idx, val_idx)
    elif experiment == "MSI_classification":
        base_dir = os.path.join(os.environ['PATHOMIX_DATA'], 'MSI_classification')
        data_dir_train = os.path.join(base_dir, 'train')
        data_dir_val = os.path.join(base_dir, 'test')
        df_train = create_data_frame(base_dir=data_dir_train)
        df_val = create_data_frame(base_dir=data_dir_val)

    vis_dir = os.path.join(base_dir, 'visualize')

    if data_gen_dict['do_augmentation']:
        print('create data generators')
        train_datagen = ImageDataGenerator(
            featurewise_center=data_gen_dict["featurewise_center"],
            samplewise_center=data_gen_dict["samplewise_center"],
            featurewise_std_normalization=data_gen_dict["featurewise_std_normalization"],
            samplewise_std_normalization=data_gen_dict["samplewise_std_normalization"],
            rotation_range=data_gen_dict["rotation_range_train"],
            width_shift_range=data_gen_dict["width_shift_range_train"],
            height_shift_range=data_gen_dict["height_shift_range_train"],
            horizontal_flip=data_gen_dict["horizontal_flip_train"],
            vertical_flip=data_gen_dict["vertical_flip_train"],
            fill_mode=data_gen_dict["fill_mode_train"],
            cval=data_gen_dict["cval_train"]
        )
        val_datagen = ImageDataGenerator(
            featurewise_center=data_gen_dict["featurewise_center"],
            samplewise_center=data_gen_dict["samplewise_center"],
            featurewise_std_normalization=data_gen_dict["featurewise_std_normalization"],
            samplewise_std_normalization=data_gen_dict["samplewise_std_normalization"],
            rotation_range=data_gen_dict["rotation_range_val"],
            width_shift_range=data_gen_dict["width_shift_range_val"],
            height_shift_range=data_gen_dict["height_shift_range_val"],
            horizontal_flip=data_gen_dict["horizontal_flip_val"],
            vertical_flip=data_gen_dict["vertical_flip_val"],
            fill_mode=data_gen_dict["fill_mode_val"],
            cval=data_gen_dict["cval_val"],
            preprocessing_function=random_crop
        )
        # train_datagen.standardize()

        print('create training batch generators')
        train_generator = train_datagen.flow_from_dataframe(df_train, data_dir_train,
                                                            x_col=data_gen_dict["x_col"],
                                                            y_col=data_gen_dict["y_col"],
                                                            weight_col=None,
                                                            target_size=hp_dict["input_size"],
                                                            class_mode=data_gen_dict["class_mode"],
                                                            batch_size=hp_dict["batch_size_ul"],
                                                            shuffle=True,
                                                            seed=hp_dict["seed"],
                                                            save_to_dir=None,
                                                            save_prefix="aug_test_")

        val_generator = val_datagen.flow_from_dataframe(df_val, data_dir_val,
                                                        x_col=data_gen_dict["x_col"],
                                                        y_col=data_gen_dict["y_col"],
                                                        weight_col=None,
                                                        target_size=hp_dict["input_size"],
                                                        class_mode=data_gen_dict["class_mode"],
                                                        batch_size=hp_dict["batch_size_ul"],
                                                        shuffle=True,
                                                        seed=hp_dict["seed"],
                                                        save_to_dir=None,
                                                        save_prefix="aug_test_val")

        # for hyperparameter tracking
        devide_by = data_gen_dict['devide_by']
        labels = list(train_generator.class_indices.keys())
        step_per_epoch = len(train_generator) // devide_by
        validation_steps = len(val_generator)
    else:
        devide_by = data_gen_dict['devide_by']
        labels = list(df_total['label'].unique())
        params = {
            'batch_size_ul': hp_dict["batch_size_ul"],
            'dim': hp_dict['input_size'],
            'n_channels': 3,
            'shuffle': True
        }
        train_generator = DataLoader(data_frame=df_train, data_dir=data_dir_train, **params)
        step_per_epoch = train_generator.__len__() // devide_by
        val_generator = DataLoader(data_frame=df_val, data_dir=data_dir_val, **params)
        validation_steps = val_generator.__len__()

    # add the retrieved values to wandb logs later on
    property_dict = dict(
        labels=labels,
        step_per_epoch=step_per_epoch,
        validation_steps=validation_steps,
        max_queue_size=multiprocessing.cpu_count() * 3,
        workers=multiprocessing.cpu_count(),
    )

    print("load model")
    # load model with pretrained- weights
    if debug:
        model = efn.EfficientNetB0(weights='imagenet', include_top=False)
    else:
        model = efn.EfficientNetB5(weights='imagenet', include_top=False)

    # freeze all layers in pretrained model
    for l in model.layers:
        l.trainable = False

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.2)(x)
    pred = Dense(len(property_dict["labels"]), activation='softmax')(x)

    my_model = Model(inputs=model.input, outputs=pred)
    print('complile model')
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    if hp_dict['optimizer'] == 'sgd':
        sgd = optimizers.SGD(learning_rate=optimizing_parameters["lr"], momentum=optimizing_parameters["momentum"],
                             nesterov=hp_dict["nesterov"], decay=optimizing_parameters["decay"])
        my_model.compile(optimizer=sgd, loss=hp_dict["loss"], metrics=hp_dict["metrics"])
    elif hp_dict['optimizer'] == 'rmsprop':
        my_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=hp_dict["metrics"])
    else:
        print("invalid optimizer")

    '''
    wandb_callback = WandbCallback(monitor='val_loss', mode='max', save_weights_only=False, log_weights=False,
                                   log_gradients=False, save_model=False, training_data=None,
                                   validation_data=None, labels=hp_dict["labels"], data_type="image", predictions=32,
                                   generator=val_generator)
    '''
    wandb_callback = WandbCallback(monitor='val_loss', mode='max', save_weights_only=False, log_weights=False,
                                   log_gradients=False, save_model=False, training_data=None,
                                   validation_data=None, labels=None, data_type=None, predictions=0,
                                   generator=val_generator)

    paras_dict_determined = {**data_gen_dict, **hp_dict}
    all_paras_dict_determined = {**paras_dict_determined, **property_dict}
    wandb.config.update(params=all_paras_dict_determined)

    print('start training')
    if data_gen_dict['do_augmentation']:
        my_model.fit_generator(train_generator, steps_per_epoch=property_dict["step_per_epoch"],
                               epochs=hp_dict["epochs_ul"], verbose=hp_dict["verbose"], callbacks=[wandb_callback],
                               validation_data=val_generator, validation_steps=property_dict["validation_steps"],
                               validation_freq=hp_dict["validation_freq"], class_weight=hp_dict["class_weight"],
                               max_queue_size=property_dict["max_queue_size"], workers=property_dict["workers"],
                               use_multiprocessing=hp_dict["use_multiprocessing"], shuffle=hp_dict["shuffle"],
                               initial_epoch=hp_dict[
                                   "initial_epoch"])  # (x=train_generator, epochs=callbacks=[WandbCallback()])
    else:
        my_model.fit(train_generator.data, to_categorical(train_generator.labels),
                     batch_size=hp_dict["batch_size_ul"],
                     epochs=hp_dict["epochs"],
                     validation_data=(val_generator.data, to_categorical(val_generator.labels)),
                     shuffle=True)

    #
    # start fine tuning
    #
    for layer in my_model.layers:
        layer.trainable = True
    sgd = optimizers.SGD(learning_rate=optimizing_parameters["lr"], momentum=optimizing_parameters["momentum"],
                         nesterov=hp_dict["nesterov"], decay=optimizing_parameters["decay"])
    my_model.compile(optimizer=sgd, loss=hp_dict["loss"], metrics=hp_dict["metrics"])

    train_generator = train_datagen.flow_from_dataframe(df_train, data_dir_train,
                                                        x_col=data_gen_dict["x_col"],
                                                        y_col=data_gen_dict["y_col"],
                                                        weight_col=None,
                                                        target_size=hp_dict["input_size"],
                                                        class_mode=data_gen_dict["class_mode"],
                                                        batch_size=hp_dict["batch_size_ft"],
                                                        shuffle=True,
                                                        seed=hp_dict["seed"],
                                                        save_to_dir=None,
                                                        save_prefix="aug_test_")

    my_model.fit_generator(train_generator, steps_per_epoch=property_dict["step_per_epoch"],
                           epochs=hp_dict["epochs_ft"], verbose=hp_dict["verbose"], callbacks=[wandb_callback],
                           validation_data=val_generator, validation_steps=property_dict["validation_steps"],
                           validation_freq=hp_dict["validation_freq"], class_weight=hp_dict["class_weight"],
                           max_queue_size=property_dict["max_queue_size"], workers=property_dict["workers"],
                           use_multiprocessing=hp_dict["use_multiprocessing"], shuffle=hp_dict["shuffle"],
                           initial_epoch=hp_dict["initial_epoch"])

    if cf.model_out_name:
        my_model.save(os.path.join(wandb.run.dir, cf.model_out_name))
