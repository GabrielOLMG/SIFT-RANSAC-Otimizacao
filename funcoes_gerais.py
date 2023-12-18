import os
import cv2 as cv
import pandas as pd

def open_image(image_path):
    return cv.imread(image_path,0)


def open_images(imovel_path):
    list_of_images_name = os.listdir(imovel_path)
    dict_of_images = {}

    for image_name in list_of_images_name:
        image_path = os.path.join(imovel_path, image_name)
        image = open_image(image_path)
        dict_of_images[image_path] = image
    
    return dict_of_images



    