# code adopted form
# https://github.com/gerstung-lab/PC-CHiP/blob/master/inception/preprocess/imgconvert.py

import os
import sys
import numpy as np
#import cv2
from openslide import OpenSlide


def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag


def main():
    #filepath = sys.argv[1]
    debug = True
    filename = 'TCGA-AA-3562-01Z-00-DX1.e07893e6-646d-41b5-be51-9c19d51f6743.svs'
    data_root = os.environ['PATHOMIX_DATA']
    input = os.path.join(data_root, 'WSI')
    output = os.path.join(data_root, 'output')
    tile_size_x = 224
    tile_size_y = 224
    # add more code before changing this parameter
    zoom_level = 0

    filepath = os.path.join(input, filename)
    img = OpenSlide(filepath)

    if debug:
        x_0 = 40000
        y_0 = 40000
        w = 41000
        h = 41000
    else:
        x_0 = y_0 = 0
        [w, h] = img.dimensions

    for x in range(x_0, w, tile_size_x):
        for y in range(y_0, h, tile_size_y):
            img_reg = img.read_region(location=(x,y), level=zoom_level, size=(tile_size_x, tile_size_y))
            img_converted = img_reg.convert('RGB') # otherwise RGBA
            out_filename = os.path.join(output, "{}_{}_{}.png".format(filename, x, y))
            img_converted.save(out_filename)


if __name__ == '__main__':
    main(sys.argv[1:])