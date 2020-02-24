import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from PIL import Image
import cv2

def load_img(path):
    return np.float64(imageio.imread(path))


def normalize_total(x):
    x -= np.mean(x, keepdims=True)
    return x

def create_histogram_total(img, out_path):
    plt.hist(img.flatten(), 255)
    plt.savefig(out_path)
    plt.close('all')
    return None

def create_histo_channel(img, out_path):
    for i in range(img.shape[-1]):
        plt.hist(img[:,:,i].flatten(), 255)
    plt.savefig(out_path)
    plt.close('all')
    return None


def create_out_path(out_dir, file_name):
    file_name += '.png'
    return os.path.join(out_dir, file_name)


def show_np(np_arr):
    img = Image.fromarray(np_arr, 'RGB')
    img.show()
    return None

def save_np(np_arr, out_path):
    img = Image.fromarray(np_arr, 'RGB')
    img.save(out_path)
    return None


def save_np_batch(batch, out_dir):
    for idx in range(batch.shape[0]):
        fn = create_out_path(out_dir, 'img_{}'.format(str(idx).zfill(2)))
        save_np(batch[idx].astype('uint8'), fn)