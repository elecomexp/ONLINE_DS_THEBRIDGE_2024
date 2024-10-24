"""
vizdatatools.py

Author: Lander Combarro Exposito
Created: 2024/10/23
Last Modified: 2024/10/23

Description
-----------
This module provides a set of functions for visualizing data using popular 
Python libraries like Matplotlib and Seaborn.

Functions
---------
>>> plot_multiple_categorical_distributions()
>>> plot_multiple_histograms_KDEs_boxplots()
>>> plot_categorical_numerical_relationship()                           
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


'''
################################################################################################### 
#    UNIVARIATE Analysis                UNIVARIATE Analysis                 UNIVARIATE Analysis   #
###################################################################################################
'''

def plot_multiple_categorical_distributions(df, categorical_columns, *, relative=False, show_values=True, rotation=45, palette='viridis') -> None:
    '''
    Plot a bar-graphs matrix, with 2 columns and the rows needed to plot the
    `absolute` or `relative` frequency from the categorical columns of `df`.
    
    Parameters
    ----------
    df : pandas.DataFrame
        pandas.DataFrame

    categorical_columns : list
        Categorical columns   

    relative : bool, optional
        If True, it plots Relative Frecuency.
        
    show_values : bool, optional
        If True, show numerical values over each bar.
        
    rotation : int, optional
        X-Tick label rotation.
        
    palette : None, palette name, list, or dict, optional
        Colors to use for the different levels of the hue variable. 
        Should be something that can be interpreted by color_palette(), 
        or a dictionary mapping hue levels to matplotlib colors. 
    '''
    num_columns = len(categorical_columns)
    num_rows = (num_columns // 2) + (num_columns % 2)

    if num_columns == 1:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    else:
        fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
        axs = axs.flatten()     # Return a copy of the array collapsed into one dimension.

    for i, col in enumerate(categorical_columns):
        if num_columns > 1:
            ax = axs[i]
        if relative:    
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x = serie.index, y = serie, ax = ax, palette = palette, hue = serie.index, legend = False)
            ax.set_ylabel('Relative Frequency')
        else:
            serie = df[col].value_counts()
            sns.barplot( x = serie.index, y = serie, ax = ax, palette = palette, hue = serie.index, legend = False)
            ax.set_ylabel('Frecuency')

        ax.set_title(f'{col}: Distribution')
        ax.set_xlabel('')
        ax.tick_params(axis = 'x', rotation = rotation)

        if show_values:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_rows * 2):
        if num_columns > 1:
            axs[j].axis('off')

    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_multiple_histograms_KDEs_boxplots(df, columns, *, kde=True, boxplot=True, whisker_width=1.5, bins=None) -> None:
    '''
    Plot histogram, KDE and Box-Plots in one figure, using `plt.subplots()` and `"Seaborn"`
    
    Parameters
    ----------
    df : pandas.DataFrame
        `pandas.DataFrame` to evaluate.
    
    columns : list
        Numerical columns from `df`

    kde : bool, optional
        If True, plot the KDE. Default is True.

    boxplot : bool, optional
        If True, plot the boxplot. Default is True.
                
    whisker_width : float, optional
        Width of the whiskers. Default is 1.5.
    
    bins : None or str, number, vector, or a pair of such values, optional
        Number of bins for the groups. Default is "auto".
    '''
    num_columns = len(columns)
    if num_columns:
        if boxplot:
            fig, axs = plt.subplots(num_columns, 2, figsize=(12, 5 * num_columns))
        else:
            fig, axs = plt.subplots(num_columns, 1, figsize=(6, 5 * num_columns))

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histogram and KDE
                sns.histplot(df[column], kde=kde, ax=axs[i, 0] if boxplot and num_columns > 1 else axs[i], bins="auto" if not bins else bins[i])
                if boxplot:
                    if kde:
                        axs[i, 0].set_title(f'{column}: Histogram and KDE')
                    else:
                        axs[i, 0].set_title(f'{column}: Histogram')
                else:
                    if kde:
                        axs[i].set_title(f'{column}: Histogram and KDE')
                    else:
                        axs[i].set_title(f'{column}: Histogram')

                # Boxplot
                if boxplot:
                    sns.boxplot(x=df[column], ax=axs[i, 1] if num_columns > 1 else axs[i + num_columns], whis=whisker_width)
                    axs[i, 1].set_title(f'{column}: BoxPlot')

        plt.tight_layout()
        plt.show()


'''
###################################################################################################
#     BIVARIATE Analysis                 BIVARIATE Analysis                  BIVARIATE Analysis   #
###################################################################################################
'''

'''
##############################################
#   Two Variables: CATEGORICAL + NUMERICAL   #
##############################################
'''

def plot_categorical_numerical_relationship(df, categorical_col, numerical_col, show_values=True, measure='mean', group_size=5):
    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending=False)

    # Si hay más de group_size categorías, las divide en grupos de group_size
    if grouped_data.shape[0] > group_size:
        unique_categories = grouped_data.index.unique()
        num_plots = int(np.ceil(len(unique_categories) / group_size))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * group_size:(i + 1) * group_size]
            data_subset = grouped_data.loc[categories_subset]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=data_subset.index, y=data_subset.values)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {categorical_col} y {numerical_col} - Grupo {i + 1}')
            plt.xlabel(categorical_col)
            plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de group_size categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=grouped_data.index, y=grouped_data.values)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {categorical_col} y {numerical_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.grid()
        plt.show()
