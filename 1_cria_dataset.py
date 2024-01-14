'''
    O objetivo vai ser entrar no site https://fotos.confidencialimobiliario.com/ e baixar fotos de imoveis, para depois eu ver se tem imagens semelhantes e fazer a analise.
    Passos:

'''

### INICIO VARIAVEIS GLOBAIS

URL = 'https://fotos.confidencialimobiliario.com/admin/freddy/imovel/?id=306307'
USERNAME = 'goliveira'
PASSWORD = 'Mosegaw_999'

### FIM VARIAVEIS GLOBAIS

import os
import time
import urllib.request

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

def get_drive():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--log-level=3")  # Configura o nível de log para "OFF" (nenhum log)

    driver = webdriver.Chrome(options=chrome_options)

    return driver

def escrever_input(driver, find_by_type, name, input_, send=True, time_wait=2):
    input_text = driver.find_element(find_by_type, name)
    input_text.send_keys(input_)

    if send:
        input_text.send_keys(Keys.RETURN)
    if time_wait:
        time.sleep(time_wait)
    return input_text

def clica_botao(driver, find_by_type, name,time_wait=2):
    button_choose_name = driver.find_element(find_by_type, name)
    button_choose_name.click()

    time.sleep(time_wait)


def logging(driver):
    escrever_input(driver, By.NAME, "username", USERNAME, send=False, time_wait=0)
    
    escrever_input(driver, By.NAME, "password", PASSWORD, send=True)

def process_image(driver, url, id_imovel, image_number):
    driver.get(url)
    div_element = driver.find_element(By.CLASS_NAME, "fieldBox.field-imagem_original")
    a_element = div_element.find_element(By.TAG_NAME, "a")

    # Obtenha o valor do atributo "href" do elemento <a>
    href = a_element.get_attribute("href")

    path_to_download = os.path.join("data_teste_agrupa_semelhantes",id_imovel)
    os.makedirs(path_to_download, exist_ok=True)
    name_file = f"{image_number}_{id_imovel}.jpg"
    image_path = os.path.join(path_to_download, name_file)
    urllib.request.urlretrieve(href, image_path)
    time.sleep(1)



def process_imovel(driver, id_imovel):
    # Entra No Imovel
    clica_botao(driver, By.LINK_TEXT, id_imovel, time_wait=1)

    # Pega o ids das Imagens
    list_of_links = get_list(driver,By.CLASS_NAME, "inlinechangelink", "href")
    
    for i,image_link in enumerate(list_of_links):
        print(f"\t\t {i} de {len(list_of_links)}: ")

        process_image(driver, image_link, id_imovel,i)
    #time.sleep(2)

    # Volta Para A Pagina Anterior 26
    driver.get(URL)

def get_list(driver,by_type, type_, att):
    elements = driver.find_elements(by_type, type_)
    ids = []

    for element in elements:
        if att == "text":
            value = element.text
        elif att == "href":
            value = element.get_attribute("href")
        ids.append(value)
    
    return ids


def make_web_scraping(driver):
    logging(driver)
    list_of_ids = get_list(driver,By.CSS_SELECTOR, "th.field-id a", "text")
    for i, id_ in enumerate(list_of_ids):
        print(f"{i} de {len(list_of_ids)}")
        process_imovel(driver, id_)
        
if __name__ == "__main__":
    driver = get_drive()
    driver.get(URL)
    time.sleep(1)
    make_web_scraping(driver)
