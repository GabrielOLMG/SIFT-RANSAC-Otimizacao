import os
import cv2 as cv

from funcoes_gerais import open_images
from funcoes_semelhanca import get_keypoints_descriptors, compare_keypoints_descriptors

##########
# global #
##########
PATH = "3_data_teste_agrupa_semelhantes"
SIFT_MODEL = cv.SIFT_create(nOctaveLayers=6)

if __name__ == "__main__":
    lista_de_imoveis = ["239729_menor"]    

    for i, imovel in enumerate(lista_de_imoveis):
        print("-----------------------------------")
        print(f"IMOVEL: {imovel}")
        path_imovel = os.path.join(PATH, imovel)

        imovel_images = open_images(path_imovel)
        imovel_keypoints_descriptors_images = get_keypoints_descriptors(SIFT_MODEL, imovel_images)
        dict_comp = compare_keypoints_descriptors(imovel_keypoints_descriptors_images, imovel_keypoints_descriptors_images, verbose=True)
        print(dict_comp)
        print(dict_comp["2_306307.jpg"])




        print("-----------------------------------")
