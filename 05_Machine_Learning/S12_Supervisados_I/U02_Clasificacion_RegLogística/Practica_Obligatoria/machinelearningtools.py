import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_curve, auc

def show_regression_coefs(model_reg):
    '''
    Visualizes the coefficients of a fitted regression model, including binary and multiclass logistic regression models.

    This function creates horizontal bar plots of the regression model's coefficients.
    For multiclass models, it creates a subplot for each class showing the original coefficients,
    and another subplot showing the absolute values of the coefficients sorted in ascending order. 
    Additionally, it returns a DataFrame containing the coefficients.

    Parameters
    ----------
    model_reg : sklearn.base.ClassifierMixin or sklearn.base.RegressorMixin
        A fitted scikit-learn regression model. The model should have the attributes 
        `coef_` (the model's coefficients) and `feature_names_in_` (the names of the features).
    
    Returns
    -------
    df_coef : pandas.DataFrame
        A DataFrame containing the coefficients of the regression model. For multiclass models,
        the DataFrame has one column per class. The index consists of the feature names.
    
    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> model_logreg = LogisticRegression(multi_class='multinomial').fit(X_train, y_train)
    >>> show_regression_coefs(model_logreg)
    
    Notes
    -----
    This function is designed to handle both binary and multiclass logistic regression models.
    '''
    # Handle binary or multiclass coefficients
    if model_reg.coef_.shape[0] > 1:  # Multiclass case
        coef = model_reg.coef_.T  # Transpose to have features as rows, classes as columns
        columns = [f"Class {i}" for i in range(coef.shape[1])]
    else:  # Binary case
        coef = model_reg.coef_.flatten()  # Flatten to 1D array
        columns = ["coefs"]

    # Create DataFrame with coefficients
    df_coef = pd.DataFrame(coef, index=model_reg.feature_names_in_, columns=columns)
    
    # Determine the number of columns for subplots
    num_cols = len(columns)
    
    # Plotting the coefficients
    fig, axs = plt.subplots(2, num_cols, figsize=(15, 10))
    fig.suptitle(f"Regression Model Coefficients - {str(model_reg)}")

    # Ensure axs is a 2D array even if num_cols == 1
    if num_cols == 1:
        axs = axs.reshape(2, 1)

    for i, col in enumerate(df_coef.columns):
        df_coef[col].plot(kind="barh", ax=axs[0, i], legend=False, title=f"Original Coefs: {col}")
        df_coef[col].abs().sort_values().plot(kind="barh", ax=axs[1, i], legend=False, title=f"Absolute Coefs: {col}")

    fig.tight_layout()
    plt.show()

    return df_coef




# def show_regression_coefs(model_reg):
#     '''
#     Visualizes the coefficients of a fitted regression model, including multiclase logistic regression models.

#     This function creates horizontal bar plots of the regression model's coefficients.
#     For multiclase models, it creates a subplot for each class showing the original coefficients,
#     and another subplot showing the absolute values of the coefficients sorted in ascending order. 
#     Additionally, it returns a DataFrame containing the coefficients.

#     Parameters
#     ----------
#     model_reg : sklearn.base.ClassifierMixin or sklearn.base.RegressorMixin
#         A fitted scikit-learn regression model. The model should have the attributes 
#         `coef_` (the model's coefficients) and `feature_names_in_` (the names of the features).
    
#     Returns
#     -------
#     df_coef : pandas.DataFrame
#         A DataFrame containing the coefficients of the regression model. For multiclase models,
#         the DataFrame has one column per class. The index consists of the feature names.
    
#     Example
#     -------
#     >>> from sklearn.linear_model import LogisticRegression
#     >>> model_logreg = LogisticRegression(multi_class='multinomial').fit(X_train, y_train)
#     >>> show_regression_coefs(model_logreg)
    
#     Notes
#     -----
#     This function is designed to handle both binary and multiclase logistic regression models.
#     '''
#     # Handle binary or multiclass coefficients
#     if len(model_reg.coef_.shape[0]) > 1:
#         coef = model_reg.coef_.T  # Transpose to have features as rows, classes as columns
#         columns = [f"Class {i}" for i in range(coef.shape[1])]
#     else:
#         coef = model_reg.coef_.flatten()
#         columns = ["coefs"]

#     # Create DataFrame with coefficients
#     df_coef = pd.DataFrame(coef, index=model_reg.feature_names_in_, columns=columns)
    
#     # Plotting the coefficients
#     fig, axs = plt.subplots(2, coef.shape[1], figsize=(15, 10))
#     fig.suptitle(f"Regression Model Coefficients - {str(model_reg)}")

#     for i, col in enumerate(df_coef.columns):
#         df_coef[col].plot(kind="barh", ax=axs[0, i], legend=False, title=f"Original Coefs: {col}")
#         df_coef[col].abs().sort_values().plot(kind="barh", ax=axs[1, i], legend=False, title=f"Absolute Coefs: {col}")

#     fig.tight_layout()
#     plt.show()

#     return df_coef


## ###############
## S12, U02, Ejercicios workout tengo funciones interesantes
## ###############


def plot_roc_curve(model, X, y):
    '''
    Plots the Receiver Operating Characteristic (ROC) curve for a given model's predictions.
    Typically apply on test set.

    Parameters:
    -----------
    model : estimator object
        The fitted model from which to predict probabilities.
    
    X : array-like of shape (n_samples, n_features)
        The input samples to predict probabilities on.
    
    y : array-like of shape (n_samples,)
        True binary labels for X.

    Returns:
    --------
    None
        This function plots the ROC curve but does not return any value.
    '''
    
    # Predict probabilities
    scores = model.predict_proba(X)
    
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y, scores[:, 1])
    roc_auc = auc(fpr, tpr)
    print("AUC: %.2f" %(roc_auc))
    
    # Plotting the ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, color='blue', label=f'ROC (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
    plt.ylabel('True Positive Rate (Recall)')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.grid()
    plt.axis([0, 1, 0, 1])
    plt.legend(loc='lower right', fontsize=13)
    plt.show()



def get_features_importance(model, X_train, top_n=5):
    '''
    Returns the top N most important features based on the absolute value of the coefficients
    from a fitted logistic regression model.

    Parameters:
    -----------
    model : estimator object
        The fitted logistic regression model (e.g., from sklearn's LogisticRegression).
    
    X_train : DataFrame
        The DataFrame containing the training features used to fit the model.
    
    top_n : int, optional, default=5
        The number of top features to return based on the absolute value of the coefficients.

    Returns:
    --------
    DataFrame
        A DataFrame containing the top N features with the highest absolute coefficients, sorted in descending order.
    '''
    
    # Extract the intercept and coefficients from the model
    intercept = model.intercept_
    coefs = model.coef_.ravel()
    
    # Create a DataFrame to hold the features and their corresponding coefficients
    features = pd.DataFrame(coefs, X_train.columns, columns=['coefficient']).copy()
    
    # Take the absolute value of the coefficients
    features['coefficient'] = np.abs(features['coefficient'])
    
    # Sort the features by the absolute value of the coefficients and return the top N
    top_features = features.sort_values('coefficient', ascending=False).head(top_n)
    
    return top_features



def plot_features_importance(model, X_train, y_train):
    '''
    Calculates and plots the standardized feature importance based on the absolute value of the coefficients
    from a fitted logistic regression model, adjusted for feature standard deviations.

    Parameters:
    -----------
    model : estimator object
        The fitted logistic regression model (e.g., from sklearn's LogisticRegression).
    
    X_train : DataFrame
        The DataFrame containing the training features used to fit the model.
    
    y_train : Series
        The Series containing the training target variable.
    
    Returns:
    --------
    None
        This function plots the standardized feature importance but does not return any value.
    '''
    
    # Extract the coefficients from the model
    coefs = model.coef_.ravel()
    
    # Create a DataFrame to hold the features and their corresponding coefficients
    features = pd.DataFrame(coefs, X_train.columns, columns=['coefficient']).copy()
    
    # Calculate the standard deviation of each feature
    stdevs = [X_train[i].std() for i in X_train.columns]
    features['stdev'] = np.array(stdevs).reshape(-1, 1)
    
    # Calculate the importance of each feature
    features['importance'] = features['coefficient'] * features['stdev']
    features['importance_standardized'] = features['importance'] / y_train.std()
    
    # Sort features by standardized importance
    features = features.sort_values('importance_standardized', ascending=True)
    
    # Plot the standardized importance of features
    plt.figure(figsize=(10, 8))
    plt.barh(features.index, features['importance_standardized'], color='skyblue')
    plt.xlabel('Standardized Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance Standardized by Coefficient and Feature Std Dev')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()



##############################################
# DEPRECATED        DEPRECATED      DEPRECATED
##############################################

def show_regression_coefs_OLD(model_reg):
    '''
    Visualizes the coefficients of a fitted regression model.

    This function creates a horizontal bar plot of the regression model's coefficients.
    It displays two subplots: one showing the original coefficients and the other
    showing the absolute values of the coefficients sorted in ascending order. 
    Additionally, it returns a DataFrame containing the coefficients.

    Parameters
    ----------
    model_reg : sklearn.base.RegressorMixin
        A fitted scikit-learn regression model. The model should have the attributes 
        `coef_` (the model's coefficients) and `feature_names_in_` (the names of the features).
    
    figsize : tuple, optional, default=(10, 5)
        The size of the figure for the bar plots. It is passed to `matplotlib.pyplot.subplots`.
    
    Returns
    -------
    df_coef : pandas.DataFrame
        A DataFrame containing the coefficients of the regression model. The index consists of the feature 
        names and the single column is labeled "coefs".
    
    Example
    -------
    >>> from sklearn.linear_model import Ridge
    >>> model_ridge = Ridge().fit(X_train, y_train)
    >>> show_coefs(model_ridge)
    
    Notes
    -----
    This function requires the regression model to have been trained on the dataset with the `coef_` 
    and `feature_names_in_` attributes, which are standard for linear models like 
    `Ridge`, `Lasso`, or `LinearRegression` in scikit-learn.
    '''
    df_coef = pd.DataFrame(model_reg.coef_, index=model_reg.feature_names_in_, columns=["coefs"])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    df_coef.plot(kind="barh", ax=ax[0], legend=False)
    df_coef.abs().sort_values(by="coefs").plot(kind="barh", ax=ax[1], legend=False)
    fig.suptitle(f"Regression Model Coefficients - {str(model_reg)}")

    fig.tight_layout()

    return df_coef