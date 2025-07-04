import logging
import os
import cv2
import numpy as np

# from utils.constants import LOGGING_NAMESPACE
from constants import LOGGING_NAMESPACE

logger = logging.getLogger(LOGGING_NAMESPACE)


def load_img_data(input_name, img_dir, data_type, width=0, height=0, count=0, pre_process_func=None):
    assert os.path.exists(img_dir)
    assert 0 <= width <= 4096
    assert 0 <= height <= 2160
    img_paths = get_total_img_paths_in_dir(img_dir)
    logger.info('img name:{}'.format((img_paths)))
    file_loaded = 0
    datas = []
    for img_path in img_paths:
        img_data = pre_process_func(img_path, data_type, width, height)
        datas.append({input_name: img_data})
        file_loaded += 1
        if file_loaded % 100 == 0:
            logger.info('{} image files have been loaded!'.format(file_loaded))
        if file_loaded == count:
            break
    logger.info('{} image files have been loaded!'.format(file_loaded))
    return datas


def load_and_resize_img(filename, data_type, width, height):
    image = cv2.imread(filename)
    if width != 0 and height != 0:
        image = cv2.resize(image, (width, height))

    # convert to nchw format and return
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    return np.expand_dims(np.array([r, g, b]), axis=0).astype(data_type)


def is_valid_img_file(img_name):
    return img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png')


def get_total_img_paths_in_dir(dir_path):
    assert os.path.exists(dir_path)
    img_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if is_valid_img_file(file):
                img_path = os.path.join(root, file)
                img_paths.append(img_path)
    return img_paths


def get_total_img_count_in_dir(dir_path):
    return len(get_total_img_paths_in_dir(dir_path))