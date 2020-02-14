import os
import pandas as pd
import shutil
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import efficientnet.keras as efn

import wandb
from wandb.keras import WandbCallback


def create_data_frame(base_dir):
    data_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.png'):
                dir_name = os.path.basename(root)
                relative_path = os.path.join('.', dir_name, file)
                data_list.append({"relative_path": relative_path, "label": os.path.basename(root)})
    data_frame = pd.DataFrame(data_list)
    return data_frame


def split_data_frame(df, train_idx, val_idx):
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    return df_train, df_val


def list_all_files_for_class(base_dir, label):
    list_of_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if not label in root:
                break
            if file.endswith('.png'):
                dir_name = os.path.basename(root)
                list_of_files.append(os.path.join(dir_name, file))
    return list_of_files


def pick_random_sample(file_list, proportion=0.2):
    pick_n = int(len(file_list) * proportion)
    return random.sample(file_list, pick_n)


def move_files(file_list, source_dir, target_dir, label):
    target_folder = os.path.join(target_dir, label)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    for f in file_list:
        source_path = os.path.join(source_dir, f)
        target_path = os.path.join(target_dir, f)
        shutil.move(source_path, target_path)


def create_test_set(base_dir=os.path.join(os.environ['PATHOMIX_DATA'], 'Jakob_cancer_detection', 'train')):
    labels = ['ADIMUC', 'STRMUS', 'TUMSTU']
    source_dir = '/home/pmf/Documents/DataMining/datasets/pathology/Jakob_cancer_detection/train'
    target_dir = '/home/pmf/Documents/DataMining/datasets/pathology/Jakob_cancer_detection/test'

    for l in labels:
        total_list = list_all_files_for_class(base_dir, l)
        random_list = pick_random_sample(total_list, proportion=0.2)
        move_files(random_list, source_dir, target_dir, l)


width_shift_range = 0.2
seed = 42
batch_size = 8
input_size = (224, 224)
# input_size = (456,456)


base_dir = os.path.join(os.environ['PATHOMIX_DATA'], 'Jakob_cancer_detection')
data_dir = os.path.join(base_dir, 'train')
vis_dir = os.path.join(base_dir, 'visualize')

train_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=True,
    featurewise_std_normalization=False,
    samplewise_std_normalization=True,
    rotation_range=90,
    width_shift_range=width_shift_range,
    height_shift_range=width_shift_range,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0
)
val_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=True,
    featurewise_std_normalization=False,
    samplewise_std_normalization=True,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='constant',
    cval=0
)

df_total = create_data_frame(base_dir=data_dir)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
# get indices for train and validation
train_idx, val_idx = next(kf.split(X=np.zeros(len(df_total)), y=df_total['label']))
df_train, df_val = split_data_frame(df_total, train_idx, val_idx)

train_generator = train_datagen.flow_from_dataframe(df_train, data_dir, x_col='relative_path', y_col='label',
                                                    weight_col=None,
                                                    target_size=input_size, class_mode='categorical',
                                                    batch_size=batch_size,
                                                    shuffle=True, seed=seed, save_to_dir=None,
                                                    save_prefix="aug_test_")
val_generator = val_datagen.flow_from_dataframe(df_val, data_dir, x_col='relative_path', y_col='label', weight_col=None,
                                                target_size=input_size, class_mode='categorical', batch_size=batch_size,
                                                shuffle=True, seed=seed, save_to_dir=None,
                                                save_prefix="aug_test_val")

epochs = 10
lr = 0.01
decay = 1e-6
momentum = 0.0
nesterov = False

labels =list(train_generator.class_indices.keys())

# load model with pretrained- weights
model = efn.EfficientNetB0(weights='imagenet')
x = model.output
pred = Dense(len(labels), activation='sigmoid')(x)

model = Model(inputs=model.input, outputs=pred)

sgd = optimizers.SGD(lr=lr, momentum=momentum, nesterov=nesterov, decay=decay)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

wandb.init(name="first_run", project="first_aws")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size

# Magic
train_loss = 0

wandb_callback = WandbCallback(monitor='val-loss', mode='max', save_weights_only=False, log_weights=False,
                               log_gradients=False, save_model=False, training_data=None,
                               validation_data=None, labels=labels, data_type='image', predictions=32,
                               generator=val_generator)
# wand_callbacks = WandbCallback(data_type="image", labels=labels)


train_loss = model.fit_generator(train_generator, steps_per_epoch=100, epochs=epochs, verbose=2, callbacks=[wandb_callback],
                                 validation_data=val_generator, validation_steps=10, validation_freq=1,
                                 class_weight=None,
                                 max_queue_size=100, workers=4, use_multiprocessing=False, shuffle=True,
                                 initial_epoch=0)  # (x=train_generator, epochs=callbacks=[WandbCallback()])
print(train_loss.params)

'''
for epoch in range(1, wandb.config.epochs + 1):
    print(epoch)
    for batch_idx in range(20): #range(len(train_generator)):
        x, y = train_generator.next()
        train_loss = model.train_on_batch(x, y, sample_weight=None, class_weight=None, reset_metrics=True)
        print(train_loss)
'''
