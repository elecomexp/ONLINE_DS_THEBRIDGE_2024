import re

import numpy as np
import pandas as pd
import urllib.request

from PIL import Image


# Suponiendo que df_train_mod es tu DataFrame
def extract_cpu_features(cpu):
    # Inicializar valores
    brand = None
    family = None
    cores = None
    frequency = None
    
    # Extraer la marca
    if 'Intel' in cpu:
        brand = 'Intel'
    elif 'AMD' in cpu:
        brand = 'AMD'
    
    # Extraer la familia (i3, i5, i7, etc.)
    match = re.search(r'(i[357])|AMD (A[6-9]|FX|A10|A12)', cpu)
    if match:
        family = match.group(0)
    
    # Extraer la frecuencia
    match = re.search(r'(\d+\.\d+|\d+)GHz', cpu)
    if match:
        frequency = float(match.group(1))
    
    return pd.Series([brand, family, cores, frequency])



def extract_screen_features(screen_resolution):
    """
    Desglosa la columna "ScreenResolution" en nuevas características que representan la resolución, 
    el tipo de panel y la presencia de la función táctil (touchscreen).
    """
    # Inicializar valores
    width = None
    height = None
    is_ips = 0
    is_retina = 0
    is_touchscreen = 0
    
    # Buscar el ancho y la altura
    match = re.search(r'(\d+)x(\d+)', screen_resolution)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
    
    # Verificar si es un IPS Panel
    if 'IPS Panel' in screen_resolution:
        is_ips = 1
    
    # Verificar si es un Retina Display
    if 'Retina Display' in screen_resolution:
        is_retina = 1

    # Verificar si tiene Touchscreen
    if 'Touchscreen' in screen_resolution:
        is_touchscreen = 1
    
    return pd.Series([width, height, is_ips, is_retina, is_touchscreen])

def extract_memory_by_type(storage, storage_type):
    """
    Función para extraer la capacidad en GB por tipo de almacenamiento
    """
    capacity = 0
    for part in storage.split('+'):
        part = part.strip()
        if storage_type in part:
            # Buscar si tiene capacidad en GB o TB usando regex
            gb_match = re.search(r'(\d+)\s*GB', part)
            tb_match = re.search(r'(\d+(\.\d+)?)\s*TB', part)
            
            if gb_match:
                capacity += int(gb_match.group(1))  # Extraer valor en GB
            elif tb_match:
                capacity += int(float(tb_match.group(1)) * 1000)  # Convertir TB a GB
    return capacity



def kaggle_checker(df_to_submit, path, sample=pd.read_csv(r'./data/sample_submission.csv')):
    """
    Esta función se asegura de que tu submission tenga la forma requerida por Kaggle.
    
    Si es así, se guardará el dataframe en un `csv` y estará listo para subir a Kaggle.
    
    Si no, LEE EL MENSAJE Y HAZLE CASO.
    
    Si aún no:
    - apaga tu ordenador, 
    - date una vuelta, 
    - enciendelo otra vez, 
    - abre este notebook y 
    - leelo todo de nuevo. 
    Todos nos merecemos una segunda oportunidad. También tú.
    """
    if df_to_submit.shape == sample.shape:
        if df_to_submit.columns.all() == sample.columns.all():
            if df_to_submit.laptop_ID.all() == sample.laptop_ID.all():
                print("You're ready to submit!")
                df_to_submit.to_csv(path, index=False) #muy importante el index = False
                urllib.request.urlretrieve("https://www.mihaileric.com/static/evaluation-meme-e0a350f278a36346e6d46b139b1d0da0-ed51e.jpg", "./img/gfg.png")     
                img = Image.open("./img/gfg.png")
                img.show()   
            else:
                print("Check the ids and try again")
        else:
            print("Check the names of the columns and try again")
    else:
        print("Check the number of rows and/or columns and try again")
        print("\nMensaje secreto de Iván y Manuel: No me puedo creer que después de todo este notebook hayas hecho algún cambio en las filas de `laptops_test.csv`. Lloramos.")