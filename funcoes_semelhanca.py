import os
import glob
import tqdm
import warnings
import cv2 as cv
import statistics
import numpy as np
import pandas as pd
from skimage.measure import ransac
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.metrics import structural_similarity as compare_ssim

##########
# SCORES #
##########

def calculateScore_1(matches_size, keypoint1_size, keypoint2_size, inliers):
    # Incluir o número de inliers no cálculo do score
    denominador = min(keypoint1_size, keypoint2_size)

    if denominador == 0:
        return 0
    inlier_ratio = inliers / denominador
    return 100 * (matches_size * inlier_ratio)

def calculateScore_2(matches_size,keypoint1_size, keypoint2_size):
    denominador = min(keypoint1_size, keypoint2_size)
    if denominador == 0:
        return 0
    return 100 * (matches_size/denominador)

##########
# GERAIS #
##########

def get_ransac_result(main_keypoints, secondary_keypoints, best_match):
    warnings.filterwarnings("ignore", category=UserWarning)
    inliers, src_pts, dst_pts, error = None, [], [], False
    src_pts = np.float32([main_keypoints[m.queryIdx] for m in best_match]).reshape(
        -1, 2
    )
    dst_pts = np.float32([secondary_keypoints[m.trainIdx] for m in best_match]).reshape(
        -1, 2
    )
    model = None
    from skimage.transform import  AffineTransform, ProjectiveTransform, PolynomialTransform, warp, _warps

    m =  AffineTransform
    try:
        model, inliers = ransac(
            (src_pts, dst_pts),
            m,  # TODO: Fazer com que esse valor possa ser variavel
            min_samples=10,  # TODO: Fazer com que esse valor possa ser variavel
            residual_threshold=3,  # TODO: Fazer com que esse valor possa ser variavel
            max_trials=90000,
        )
    except Exception as e:
        error = True
    warnings.filterwarnings("default", category=UserWarning)
    return inliers, src_pts, dst_pts, error, model

def get_flann_model():
    index_params_flann = dict(
        algorithm=3,
        table_number=500,
        key_size=100,
        multi_probe_level=5,
    )
    
    

    search_params_flann = dict(checks=200)

    flann = cv.FlannBasedMatcher(index_params_flann, search_params_flann)
    return flann

def get_match(main_descriptors, secondary_descriptors, modelo, knn_value=5, ratio_test_activate = True):
    # TODO: Fazer com que o valor de knn possa ser variavel
    matches = modelo.knnMatch(main_descriptors, secondary_descriptors, k=knn_value)

    matches = ratio_test(matches,ratio_test_activate)

    return matches

def get_best_match(best_matches_1, best_matches_2):
    topMatches = []
    for match1 in best_matches_1:
        match1QueryIndex = match1.queryIdx
        match1TrainIndex = match1.trainIdx

        for match2 in best_matches_2:
            match2QueryIndex = match2.queryIdx
            match2TrainIndex = match2.trainIdx

            if (match1QueryIndex == match2TrainIndex) and (
                match1TrainIndex == match2QueryIndex
            ):
                topMatches.append(match1)

    return topMatches

def ratio_test(matches,activate=True, constant_ratio_test=0.7 ):
    # TODO: Fazer com que constant_ratio_test seja variavel
    good = []
    for match in matches:
        distance1 = match[0].distance
        distance2 = match[1].distance
        if activate:
            if distance1 < distance2 * constant_ratio_test:
                good.append(match[0])
        else:
            good.append(match[0])

    return good

def computeSIFT(sift, image):
    return sift.detectAndCompute(image, None)

##########
# GERAIS #
##########

def get_keypoints_descriptors(sift, dict_of_images):
    dict_of_infos = {}
    for image_path, image in tqdm.tqdm(dict_of_images.items()):
        keypointTemp, descriptorTemp = computeSIFT(sift, image)
        keypointTemp = [(float(kp.pt[0]), float(kp.pt[1])) for kp in keypointTemp]
        dict_of_infos[image_path] = (keypointTemp, descriptorTemp, image)

    return dict_of_infos

def _compare_keypoints_descriptors(keypoints_descriptors_image_1, keypoints_descriptors_image_2, same=None, verbose=False):
    keypoints_1,descriptors_1, image_1 = keypoints_descriptors_image_1
    keypoints_2,descriptors_2, image_2 = keypoints_descriptors_image_2

    #modelo = cv.BFMatcher()

    modelo = get_flann_model()

    match_1 = get_match(descriptors_1, descriptors_2, modelo, ratio_test_activate=True)
    match_2 = get_match(descriptors_2, descriptors_1, modelo, ratio_test_activate=True)

    matches = get_best_match(match_1, match_2)

    inliers, src_pts, dst_pts, error, model = get_ransac_result(
            keypoints_1, keypoints_2, matches
        )

    inliers = 0 if inliers is None else inliers
    score_1 = calculateScore_1(len(matches),len(keypoints_1),len(keypoints_2), np.sum(inliers))
    score_2 = calculateScore_2(len(matches),len(keypoints_1),len(keypoints_2))
    ssim_value = None #compare_ssim(image_1, image_2)

    if verbose:
        print(f"\t\t\t{score_1} --- {score_2} --- {ssim_value}")


    if not model:
        return score_1, score_2
    '''
    if True:
        M = model.params[0:2, :] 
        image1_aligned = cv.warpAffine(image_1,M, (image_2.shape[1], image_2.shape[0]))
        # # Combine as duas imagens alinhadas
        result_image = cv.addWeighted(image1_aligned, 0.5, image_2, 0.5, 0)

        cv.imshow('Imagem Alinhada', result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    '''
    return score_1,score_2 


def compare_keypoints_descriptors(info_1_keypoints_descriptors_images, info_2_keypoints_descriptors_images, verbose=False):
    score_same1 = []
    score_n_same1 = []
    score_same2= []
    score_n_same2 = []
    dict_comp = {}
    for image_path_1,keypoints_descriptors_image_1 in info_1_keypoints_descriptors_images.items():
        name_1 = image_path_1.split("\\")[-1]
        
        dict_comp[name_1] = {}
        for image_path_2,keypoints_descriptors_image_2 in info_2_keypoints_descriptors_images.items():
            if image_path_1 == image_path_2:
                continue
            name_2 = image_path_2.split("\\")[-1]
            tipo_1 = name_1.split("_")[0]
            tipo_2 = name_2.split("_")[0]
            same=tipo_1==tipo_2
            #if not same:
            #    continue
            
            score_1, score_2 = _compare_keypoints_descriptors(keypoints_descriptors_image_1, keypoints_descriptors_image_2, same=same, verbose=False)
            
            if verbose:
                print(f"{name_1} --- {name_2} --- Ransac:{score_1} --- Geral:{score_2}")
            
            if score_1:
                dict_comp[name_1][name_2] = score_1
            if score_1 is not None:
                if same:
                    score_same1.append(score_1)
                else:
                    score_n_same1.append(score_1)
            
            if score_2 is not None:
                if same:
                    score_same2.append(score_2)
                else:
                    score_n_same2.append(score_2)
        print()
    
    print("SCORE 1(COM INLINER):")
    if score_same1:
        print("SAME:",min(score_same1), statistics.median(score_same1), max(score_same1))
    if score_n_same1:
        print("NO SAME:",min(score_n_same1), statistics.median(score_n_same1), max(score_n_same1))

    print("SCORE 2:(SEM INLINER)")
    if score_same2:
        print("SAME:",min(score_same2), statistics.median(score_same2), max(score_same2))
    if score_n_same2:
        print("NO SAME:",min(score_n_same2), statistics.median(score_n_same2), max(score_n_same2))

    return dict_comp