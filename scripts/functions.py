import argparse
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from inmoose.pycombat import pycombat_norm
from pimmslearn.sklearn.ae_transformer import AETransformer


#from jinja2 import Template
#from weasyprint import HTML

# === Preprocessing Functions ===
def load_data(data_path, metadata_path):
    data = pd.read_csv(data_path, sep='\t')
    metadata = pd.read_csv(metadata_path, sep='\t')
    return data, metadata

def remove_low_quality(data, missing_feature_thresh=1, missing_sample_thresh=0.4):
    data = data.loc[:, data.isnull().mean() < missing_feature_thresh]
    data = data.loc[data.isnull().mean(axis=1) < missing_sample_thresh, :]
    return data

def impute_pimms(data):
    # use the Denoising or Variational Autoencoder
    model = AETransformer(
        model='DAE', # or 'VAE'
        hidden_layers=[512,],
        latent_dim=data.shape[0], # dimension of joint sample and item embedding
        batch_size=10,
    )
    model.fit(data,
            cuda=False,
            epochs_max=20,
            )
    return model.transform(data)

def impute_missing(data, method='knn'):
    # remove all missing features 
    data = data.dropna(axis=1, how='all')
    if method == 'knn':
        imputer = KNNImputer(n_neighbors=10)
        return pd.DataFrame(imputer.fit_transform(data), index=data.index, columns=data.columns)
    elif method == 'min':
        return data.fillna(data.min().min())
    elif method == 'pimms':
        return impute_pimms(data)
    else:
        raise ValueError("Unsupported imputation method")

def log_transform(data, pseudo_count, method='log2'):
    if method == 'log10':
        return np.log10(data + pseudo_count)
    elif method == 'log2':
        return np.log2(data + pseudo_count)
    else:
        raise ValueError("Unsupported log transformation method")
    

def normalize(data, method='median'):
    if method == 'median':
        return data.divide(data.median(axis=1), axis=0)
    elif method == 'quantile':
        transformer = QuantileTransformer(output_distribution='normal', copy=True)
        return pd.DataFrame(transformer.fit_transform(data), index=data.index, columns=data.columns)
    else:
        raise ValueError("Unsupported normalization method")

def scale(data, method='zscore'):
    if method == 'zscore':
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    elif method == 'pareto':
        return (data - data.mean()) / np.sqrt(data.std())
    else:
        raise ValueError("Unsupported scaling method")

def remove_batch_effect(data, metadata, batch_col='plate'):
    # remove features with very low variance, otherwise pycombat_norm will fail
    data = data.loc[:, data.var() > 1e-9]
    batch = metadata.loc[data.index, batch_col]
    if len(set(batch)) < 2:
        print("Warning: Batch has only 1 level, no batch correction is done")
        return data
    else:
        data_corrected = pycombat_norm(counts=data.T, batch=batch).T
        return pd.DataFrame(data_corrected, index=data.index, columns=data.columns)
