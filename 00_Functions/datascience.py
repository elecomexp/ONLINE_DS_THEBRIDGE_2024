import pandas as pd

import scipy.stats


def get_cardinality_class(df_in, umbral_categoria, umbral_continua):
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


scipy.stats.mannwhitneyu()

scipy.stats.f_oneway()