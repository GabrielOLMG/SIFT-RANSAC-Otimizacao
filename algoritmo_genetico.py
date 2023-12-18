import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
import math
from itertools import product, permutations
import concurrent.futures
import statistics

from funcoes_gerais import open_images, open_image
from funcoes_semelhanca import compare_keypoints_descriptors, computeSIFT, _compare_keypoints_descriptors

PATH = "data_real"
VAR_TYPES = {"sigma": float, "contrastThreshold": float, "nOctaveLayers": int, "edgeThreshold": int, "nfeatures": int}
VAR_MUTATION = {"sigma": 0.3, "contrastThreshold": 0.3, "nOctaveLayers": 2, "edgeThreshold": 3, "nfeatures": 2}
POPULATION_SIZE = 10

path_imagem_1 = "data_real/32972/imovel_1/1202860#009.jpg"
path_imagem_2 = "data_real/32972/imovel_2/125621013-61#022.jpg"

imagem_1 = open_image(path_imagem_1)
imagem_2 = open_image(path_imagem_2)

def view_image(image):
    imagem_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)
    plt.axis('off')  # Desativar eixos
    plt.show()


def get_keypoints_descriptors(sift, image):
    keypoint, descriptor = computeSIFT(sift, image)
    
    keypoint = [(float(kp.pt[0]), float(kp.pt[1])) for kp in keypoint]
    if len(keypoint) <=1:
        return None
    descriptor = descriptor.astype(np.float32)

    return [keypoint, descriptor, image]

def evaluates_performance(parametro, imagem_1, imagem_2):
    sift = cv.SIFT_create(**parametro)

    keypoints_descriptors_image_1 = get_keypoints_descriptors(sift, imagem_1)
    keypoints_descriptors_image_2 = get_keypoints_descriptors(sift, imagem_2)

    if (keypoints_descriptors_image_1 is None or keypoints_descriptors_image_2 is None):
        return -1, -1

    score_1_ransac,score_2_normal  =  _compare_keypoints_descriptors(keypoints_descriptors_image_1, keypoints_descriptors_image_2, same=True)
    
    return score_1_ransac,score_2_normal

def generates_mutation(parametro_nome, old_value):
    variation = random.uniform(-VAR_MUTATION[parametro_nome], VAR_MUTATION[parametro_nome])

    if VAR_TYPES[parametro_nome] == int:
        variation = round(variation)

    if parametro_nome == "nOctaveLayers":
        new_value = max(2, old_value + variation)
    else:
        new_value = max(0, old_value + variation)

    return new_value

def generate_random_population(population_size):
    population = []

    while len(population) < population_size:
        parameters = {
            "sigma": random.uniform(0.1, 1.0),
            "contrastThreshold": random.uniform(0.1, 0.5),
            "nOctaveLayers": random.randint(2, 6),
            "edgeThreshold": random.randint(5, 20),
            "nfeatures": random.randint(50, 500),
        }

        # Verifica se o conjunto de parâmetros já está na população
        if parameters not in population:
            population.append(parameters)

    return population

def evaluate_individual(individual):
    # Avaliar o desempenho do RANSAC e do método básico para um indivíduo
    score_ransac, score_basic = evaluates_performance(individual, imagem_1, imagem_2)

    return score_ransac, score_basic, individual

def evaluate_population(population):
    if False:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Avalia cada indivíduo na população em paralelo
            results = list(executor.map(evaluate_individual, population))
    results = []
    for p in population:
        results.append(evaluate_individual(p))
    
    return results

def crossover(parent1, parent2, crossover_point=None):
    if crossover_point is None:
        crossover_point = random.randint(1, len(parent1) - 1)

    child1 = {}
    child2 = {}

    keys = list(parent1.keys())

    for i in range(crossover_point):
        child1[keys[i]] = parent1[keys[i]]
        child2[keys[i]] = parent2[keys[i]]

    for i in range(crossover_point, len(parent1)):
        child1[keys[i]] = parent2[keys[i]]
        child2[keys[i]] = parent1[keys[i]]

    return [child1, child2]

def generate_population_based_on_best(best_params, population_size, used_params):
    new_population = []
    i = 0
    while len(new_population) < (population_size - 1) and i < 20:
        mutated_params = mutate_params(best_params)

        # Verifica se os parâmetros já foram utilizados
        param_tuple = tuple(mutated_params.items())
        if param_tuple not in used_params:
            new_population.append(mutated_params)
            used_params.add(param_tuple)
    if i >20:
        print("NÃO FOI POSSIVEL CRIAR UMA POPULAÇÃO TOTALMENTE NOVA")
    return new_population

def mutate_params(individual_param):
     return {param: generates_mutation(param, value) for param, value in individual_param.items()}

if __name__ == "__main__":
    geracoes = 10
    populations = generate_random_population(POPULATION_SIZE)
    avaliacao_best = []
    media_populacao = []
    tipo_1=False
    tipo_2=True
    if tipo_1:
        for geracao in range(geracoes):
            print(f"GERACAO {geracao+1} --------------------")
            print(f"\t\t TOTAL DE INDIVIDUOS: {len(populations)}")

            population_result = evaluate_population(populations)
            population_sorted = sorted(population_result, key=lambda tupla: tupla[1], reverse=True)
            best_individual_score_ransac, best_individual_score_basic, best_individual_param = population_sorted[0]

            mutated_individual = {param: generates_mutation(param, value) for param, value in best_individual_param.items()}
            new_population_crossover  = crossover(best_individual_param, mutated_individual)
            population_sorted = [population[2] for population in population_sorted]
            if len(population_sorted) >=10:
                population_sorted = population_sorted[:-3]
                population_sorted.append(mutated_individual)
            else:
                population_sorted[-1] = mutated_individual

            population_sorted.extend(new_population_crossover)
            avaliacao_best.append(best_individual_score_basic)
            populations = population_sorted
            print(f"\t\t SCORE DO MELHOR INDIVIDUO: {best_individual_score_basic} X {best_individual_score_ransac}")
            print(f"-------------------- GERACAO {geracao+1}\n")
    elif tipo_2:
        used_params = set()

        for geracao in range(geracoes):
            print(f"GERACAO {geracao+1} --------------------")
            print(f"\t\t TOTAL DE INDIVIDUOS: {len(populations)}")

            population_result = evaluate_population(populations)
            population_sorted = sorted(population_result, key=lambda tupla: tupla[1], reverse=True)
            best_individual_score_ransac, best_individual_score_basic, best_individual_param = population_sorted[0]
            population_sorted = [population[2] for population in population_sorted]

            mutated_individual = mutate_params(best_individual_param)
            new_population_crossover  = crossover(best_individual_param, mutated_individual)

            # Gera uma nova população com base nas características do melhor indivíduo
            new_population_based_on_best = generate_population_based_on_best(best_individual_param, POPULATION_SIZE-2, used_params)

            # Atualiza a população para a próxima geração
            populations = new_population_based_on_best + new_population_crossover + population_sorted[:2] + generate_random_population(2)

            avaliacao_best.append(best_individual_score_basic)
            print(f"\t\t SCORE DO MELHOR INDIVÍDUO: {best_individual_score_basic} X {best_individual_score_ransac}")
            print(f"\t\t PARAMETRO DO MELHOR INDIVÍDUO: {best_individual_param}")
            print(f"-------------------- GERACAO {geracao+1}\n")

    plt.plot(range(1, geracoes + 1), avaliacao_best, marker='o')
    plt.title('Evolução do Desempenho ao Longo das Iterações')
    plt.xlabel('Iteração')
    plt.ylabel('Desempenho')
    plt.savefig('grafico.png')