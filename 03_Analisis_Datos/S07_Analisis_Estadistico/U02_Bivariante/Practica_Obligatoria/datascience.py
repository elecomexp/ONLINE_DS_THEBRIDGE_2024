import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import scipy.stats

def get_cardinality_class(df_in, umbral_categoria = 10, umbral_continua = 20):
    '''
    Define qué tipo de variable es cada columna de un pandas.DataFrame en función de su cardinalidad.
    '''
    df_out = pd.DataFrame([df_in.nunique(), df_in.nunique()/len(df_in) * 100, df_in.dtypes])
    df_out = df_out.T.rename(columns = {0: "Card", 1: "%_Card", 2: "Tipo"})
    

    df_out.loc[df_out["Card"] < umbral_categoria, "Clase"] = "Categórica"    
    df_out.loc[df_out["Card"] == 2, "Clase"] = "Binaria"
    df_out.loc[df_out["Card"] >= umbral_categoria, "Clase"] ="Numérica Discreta"
    df_out.loc[df_out["%_Card"] > umbral_continua, "Clase"] = "Numérica Continua"
    
    return df_out


def mapa_calor(corr_matrix):
    '''
    Hay que introducir una matriz de correlación generada con pandas
    '''
    plt.figure(figsize=(10, 8))  # Ya lo veremos pero esto permite ajustar el tamaño de las gráficas
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                cbar=True, square=True, linewidths=.5) # el cmap es el rango de colores usado para representar "el calor"

    plt.title('Matriz de Correlación')
    plt.xticks(rotation=45)  # Rota las etiquetas de las x si es necesario
    plt.yticks(rotation=45)  # Rota las etiquetas de las y si es necesario

    plt.show()