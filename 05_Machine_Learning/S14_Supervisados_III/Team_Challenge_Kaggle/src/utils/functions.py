import re

import numpy as np
import pandas as pd
import urllib.request

from PIL import Image


def extract_gpu_features(gpu):
    """
    Desglosa la columna "Gpu" en nuevas características que representan la marca, 
    y modelo de la tarjeta gráfica   
    """
    parts = gpu.split()
    is_amd = 1 if parts[0] == 'AMD' else 0
    is_intel = 1 if parts[0] == 'Intel' else 0
    model = ' '.join(parts[1:])

    return pd.Series([is_amd, is_intel, model], index=['Gpu_isAMD', 'Gpu_isIntel', 'Gpu_Model'])


def extract_cpu_features(cpu):
    """
    Desglosa la columna "Cpu" en nuevas características que representan la marca, 
    sere, modelo y frecuencia (GHz) del procesador.
    """
    parts = cpu.split()
    brand = 1 if parts[0].lower() == 'amd' else 0
    frequency = float(parts[-1].replace('GHz', ''))
    series = ' '.join(parts[1:-2])
    model = parts[-2]
    
    return pd.Series([brand, series, model, frequency], index=['Cpu_isAMD', 'Cpu_Series', 'Cpu_Model', 'Cpu_GHz'])


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



def extract_memory_features(storage):
    """
    Función para extraer la capacidad en GB para cada tipo de almacenamiento.
    Devuelve un DataFrame con las capacidades en GB para HDD, SSD, Flash Storage y Hybrid,
    además de la columna Memory_Type con el tipo de almacenamiento y Memory_GB con la capacidad total.
    """
    # Inicializar un diccionario para almacenar las capacidades
    capacity_dict = {'HDD_GB': 0, 'SSD_GB': 0, 'Flash_Storage_GB': 0, 'Hybrid_GB': 0}
    
    # Inicializar una lista para almacenar los tipos de memoria detectados
    memory_types = set()

    for part in storage.split('+'):
        part = part.strip()
        
        # Verificar y sumar capacidades para cada tipo de almacenamiento
        for storage_type in capacity_dict.keys():
            if storage_type.replace('_GB', '').replace('_', ' ') in part:
                # Añadir tipo de almacenamiento a la lista
                memory_types.add(storage_type.replace('_GB', ''))
                
                gb_match = re.search(r'(\d+)\s*GB', part)
                tb_match = re.search(r'(\d+(\.\d+)?)\s*TB', part)

                if gb_match:
                    capacity_dict[storage_type] += int(gb_match.group(1))  # Extraer valor en GB
                elif tb_match:
                    capacity_dict[storage_type] += int(float(tb_match.group(1)) * 1000)  # Convertir TB a GB

    # Sumar las capacidades totales para Memory_GB
    total_memory_gb = sum(capacity_dict.values())
    
    # Crear la columna Memory_Type como una combinación de tipos detectados
    capacity_dict['Memory_Type'] = ', '.join(memory_types) if memory_types else 'None'
    capacity_dict['Memory_GB'] = total_memory_gb  # Añadir la capacidad total a capacity_dict
    
    return pd.Series(capacity_dict)




# def extract_memory_features(storage):
#     """
#     Función para extraer la capacidad en GB para cada tipo de almacenamiento.
#     Devuelve un DataFrame con las capacidades en GB para HDD, SSD, Flash Storage y Hybrid,
#     además de la columna Memory_Type con el tipo de almacenamiento.
#     """
#     # Inicializar un diccionario para almacenar las capacidades
#     capacity_dict = {'HDD_GB': 0, 'SSD_GB': 0, 'Flash_Storage_GB': 0, 'Hybrid_GB': 0}
    
#     # Inicializar una lista para almacenar los tipos de memoria detectados
#     memory_types = set()

#     for part in storage.split('+'):
#         part = part.strip()
        
#         # Verificar y sumar capacidades para cada tipo de almacenamiento
#         for storage_type in capacity_dict.keys():
#             if storage_type.replace('_GB', '').replace('_', ' ') in part:
#                 # Añadir tipo de almacenamiento a la lista
#                 memory_types.add(storage_type.replace('_GB', ''))
                
#                 gb_match = re.search(r'(\d+)\s*GB', part)
#                 tb_match = re.search(r'(\d+(\.\d+)?)\s*TB', part)

#                 if gb_match:
#                     capacity_dict[storage_type] += int(gb_match.group(1))  # Extraer valor en GB
#                 elif tb_match:
#                     capacity_dict[storage_type] += int(float(tb_match.group(1)) * 1000)  # Convertir TB a GB

#     # Crear la columna Memory_Type como una combinación de tipos detectados
#     capacity_dict['Memory_Type'] = ', '.join(memory_types) if memory_types else 'None'
    
#     return pd.Series(capacity_dict)


# def extract_memory_by_type(storage):
#     """
#     Función para extraer la capacidad en GB para cada tipo de almacenamiento.
#     Devuelve un DataFrame con las capacidades en GB para HDD, SSD, Flash Storage y Hybrid.
#     """
#     # Inicializar un diccionario para almacenar las capacidades
#     capacity_dict = {'HDD_GB': 0, 'SSD_GB': 0, 'Flash_Storage_GB': 0, 'Hybrid_GB': 0}
    
#     for part in storage.split('+'):
#         part = part.strip()
#         # Verificar y sumar capacidades para cada tipo de almacenamiento
#         for storage_type in capacity_dict.keys():
#             if storage_type.replace('_GB', '').replace('_', ' ') in part:
#                 gb_match = re.search(r'(\d+)\s*GB', part)
#                 tb_match = re.search(r'(\d+(\.\d+)?)\s*TB', part)

#                 if gb_match:
#                     capacity_dict[storage_type] += int(gb_match.group(1))  # Extraer valor en GB
#                 elif tb_match:
#                     capacity_dict[storage_type] += int(float(tb_match.group(1)) * 1000)  # Convertir TB a GB

#     return pd.Series(capacity_dict)


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