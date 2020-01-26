from tools import slide
from tools import filter
from tools import tiles

TISSUE_THRESHOLD = 95

'''
# first step: convert svs into (scaled down) jpg
#slide.training_slide_to_image(0)
slide.multiprocess_training_slides_to_images()
# apply filters in multiprocessing to all images in training_jpg
filter.multiprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=[0,1])
sum_tiles = tiles.summary_and_tiles(0, display=False, save_summary=True, save_data=True, save_top_tiles=False,
                                    save_above_threshold=True, threshold=TISSUE_THRESHOLD)
'''
def create_tiles_from_WSI():
    slide.multiprocess_training_slides_to_images()
    filter.multiprocess_apply_filters_to_images(save=True, display=False, html=False, image_num_list=[0,1,2,3,4,5,6,7,8,9])
    sum_files = tiles.multiprocess_filtered_images_to_tiles(display=False, save_summary=True, save_data=True,
                                                            save_top_tiles=False, html=True, image_num_list=[0,1,2,3,4,5,6,7,8,9],
                                                            save_above_threshold=True, threshold=TISSUE_THRESHOLD)

if __name__ == '__main__':
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = 999999999
    create_tiles_from_WSI()