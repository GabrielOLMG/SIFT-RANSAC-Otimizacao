'''
    IRA FAZER JOGAR UM IMOVEL CONTRA OUTRO IMOVEL PARA VER AS FOTOS SEMELHANTES
'''
import os
import cv2 as cv

from funcoes_gerais import open_images
from funcoes_semelhanca import get_keypoints_descriptors, compare_keypoints_descriptors
##########
# global #
##########
#PATH = "data_filtrado"
PATH = "data_real"
PATH_OUTPUT = "data_output" 
SIFT_MODEL = cv.SIFT_create(nOctaveLayers=8)
#SIFT_MODEL = cv.ORB_create(scoreType=cv.ORB_HARRIS_SCORE, WTA_K=4)
#SIFT_MODEL.setMaxFeatures(90000) 

if __name__ == "__main__":
    #p = {'sigma': 0.7536938498168531, 'contrastThreshold': 0.1255488944771055, 'nOctaveLayers': 2, 'edgeThreshold': 2, 'nfeatures': 122}
    #SIFT_MODEL = cv.SIFT_create(**p)
    #lista_de_imoveis = os.listdir(PATH)
    #lista_de_imoveis = ["32972"]
    lista_de_imoveis = ["52931"]
    print(lista_de_imoveis)
    for i,imovel in enumerate(lista_de_imoveis):
        print("-----------------------------------")
        print(f"IMOVEL: {imovel}")
        path_imovel = os.path.join(PATH, imovel)
        imovel_1_path = os.path.join(path_imovel, "imovel_1")
        imovel_2_path = os.path.join(path_imovel, "imovel_2")

        imovel_1_images = open_images(imovel_1_path)
        imovel_2_images = open_images(imovel_2_path)

        imovel_1_keypoints_descriptors_images = get_keypoints_descriptors(SIFT_MODEL, imovel_1_images)
        imovel_2_keypoints_descriptors_images = get_keypoints_descriptors(SIFT_MODEL, imovel_2_images)

        compare_keypoints_descriptors(imovel_1_keypoints_descriptors_images, imovel_2_keypoints_descriptors_images)
        print("-----------------------------------")