import os
import pandas as pd
import shutil

base_folder = '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/'
train_folder = os.path.join(base_folder, 'TRAIN_split')
validation_folder = os.path.join(base_folder, 'VALIDATION')

msi_folder = 'MSIMUT'
mss_folder = 'MSS'


def get_files_for_class(class_name, train_folder='/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/TRAIN'):
    root_folder = os.path.join(train_folder, class_name)
    class_files = [os.path.join(root_folder, x) for x in os.listdir(root_folder)]

    return class_files


def extract_id_from_files_names(files):
    # remove path of file
    f_cleaned = [x.split('/')[-1] for x in files]

    #get id
    f_ids = ['{}-{}'.format(x.split('-')[-5], x.split('-')[-4]) for x in f_cleaned]

    return f_ids


def get_ids_for_validation_set(pat_ids, validation_split=0.2):
    df = pd.DataFrame(pat_ids, columns=['pat_id'])

    df_freq = df['pat_id'].value_counts()

    number_of_slides_in_validation = df.size * validation_split

    sum_of_slides = 0
    ids_sampled = []

    while sum_of_slides < number_of_slides_in_validation:
        id_sampled = df_freq.sample(n=1).index[0]

        if not id_sampled in ids_sampled:
            ids_sampled.append(id_sampled)

            sum_of_slides += df_freq.loc[id_sampled]

    print('aimed for {} slides, finally got {}'.format(number_of_slides_in_validation, sum_of_slides))
    print('{} of {} patients will be in validation folder'.format(len(ids_sampled), df_freq.size))
    return ids_sampled


def move_files_for_validation_set(ids, files, validation_folder=validation_folder):
    # check that right class is assigned
    class_name = files[0].split('/')[-2]

    new_loaction = os.path.join(validation_folder, class_name)
    if not os.path.isdir(new_loaction):
        os.mkdir(new_loaction)
    for f in files:
        for i in ids:
            file_name = os.path.basename(f)
            if i in file_name:
                shutil.move(f, os.path.join(new_loaction, file_name))


def make_validation_set(class_name, train_folder=train_folder,validation_folder=validation_folder,  validation_split=0.2):
    files_class = get_files_for_class(class_name=class_name, train_folder=train_folder)

    pat_ids = extract_id_from_files_names(files_class)

    validation_ids = get_ids_for_validation_set(pat_ids, validation_split=validation_split)

    move_files_for_validation_set(ids = validation_ids, files=files_class, validation_folder=validation_folder)

    return None







