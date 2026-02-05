import argparse
import pandas as pd
import numpy as np
import os 
import shutil
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from inmoose.pycombat import pycombat_norm
from sklearn.decomposition import PCA

# === Preprocessing Functions ===
def load_data(data_path, metadata_path):
    # rows are samples, columns are features
    data = pd.read_csv(data_path, sep=None, index_col=0)
    metadata = None
    if metadata_path:
        metadata = pd.read_csv(metadata_path, sep=None, index_col=0)
    return data, metadata

def remove_low_quality(data, missing_feature_thresh=1, missing_sample_thresh=0.4):
    data = data.loc[:, data.isnull().mean() < missing_feature_thresh]
    data = data.loc[data.isnull().mean(axis=1) < missing_sample_thresh, :]
    return data

def remove_outlier_sample(data, threshold=1.5):
    protein_counts = data.notna().sum(axis=1)
    q1, q3 = protein_counts.quantile([0.25,0.75])
    lower_bound = q1 - threshold * (q3 - q1)
    data = data.loc[protein_counts >= lower_bound,:]
    return data

def impute_pimms(data, method='pimms_vae'):
    from pimmslearn.sklearn.ae_transformer import AETransformer
    # use the Denoising or Variational Autoencoder
    if method == 'pimms_vae':
        model = AETransformer(
            model='VAE', # or 'DAE'
            hidden_layers=[512,],
            #latent_dim=data.shape[0], # dimension of joint sample and item embedding
            #batch_size=10,
        )
        model.fit(data,
            cuda=False,
            epochs_max=100,
            )
        data_imputed = model.transform(data)
    elif method == 'pimms_dae':
        # use the Denoising Autoencoder
        model = AETransformer(
            model='DAE', # or 'VAE'
            hidden_layers=[512,],
            #latent_dim=data.shape[0], # dimension of joint sample and item embedding
            #batch_size=10,
        )
        model.fit(data,
            cuda=False,
            epochs_max=100,
            )
        data_imputed = model.transform(data)
    elif method == 'pimms_cft':
        # use the Collaborative Filtering Transformer
        from pimmslearn.sklearn.cf_transformer import CollaborativeFilteringTransformer
        index_name = 'Sample ID'
        column_name = 'protein group'
        value_name = 'intensity'

        data.index.name = index_name  # already set
        data.columns.name = column_name  # not set due to csv disk file format

        series = data.stack()
        series.name = value_name

        model = CollaborativeFilteringTransformer(
            target_column=value_name,
            sample_column=index_name,  # Sample ID
            item_column=column_name
            # n_factors=30, # dimension of separate sample and item embedding
            # batch_size=4096
        )
        model.fit(series, cuda=False, epochs_max=100)
        data_imputed = model.transform(series).unstack()

    return data_imputed

def impute_missing(data, method='knn', save_intermediate=False):    
    from sklearn.impute import KNNImputer
    # save current working directory
    old_dir = os.getcwd()

    try:
        # change working directory to intermediate 
        impute_dir = os.path.join(os.getcwd(), 'intermediate')
        if not os.path.exists(impute_dir):
            os.makedirs(impute_dir, exist_ok=True)
        os.chdir(impute_dir)
        
        # remove all missing features 
        data = data.dropna(axis=1, how='all')
        
        # do imputation
        if method == 'knn':
            imputer = KNNImputer(n_neighbors=10)
            result = pd.DataFrame(imputer.fit_transform(data), index=data.index, columns=data.columns)
        elif method.startswith('pimms'):
            result = impute_pimms(data, method=method)
        else:
            raise ValueError("Unsupported imputation method")
    finally:
        os.chdir(old_dir)

        # save intermediate results if required
        if not save_intermediate:
            shutil.rmtree(impute_dir, ignore_errors=True)

    # return final result
    return result
    
    
def log_transform(data, pseudo_count, method='log2'):
    if method == 'log10':
        return np.log10(data + pseudo_count)
    elif method == 'log2':
        return np.log2(data + pseudo_count)
    else:
        raise ValueError("Unsupported log transformation method")
    
def normalize(data, method='median'):
    # rows are samples, columns are features
    if method == 'median':
        # Compute per-sample medians
        sample_medians = data.median(axis=1)

        # Compute global median (median of medians)
        global_median = sample_medians.median()

        # Compute scaling factors (so each sample is brought to global median)
        scaling_factors = global_median / sample_medians

        # Apply normalization
        data_median_norm = data.multiply(scaling_factors, axis=0)

        return data_median_norm
    
    elif method == 'quantile':
        transformer = QuantileTransformer(output_distribution='normal', copy=True)
        return pd.DataFrame(transformer.fit_transform(data), index=data.index, columns=data.columns)
    else:
        raise ValueError("Unsupported normalization method")


def normalize(data, method='median'):
    if method == 'median':
        return data.div(data.median(axis=1), axis=0)
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


def plot_pca(data, metadata, batch_col='plate', save_file='', title=''):
    metadata = metadata.loc[data.index,:]
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.fillna(0)) 
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=data.index)
    pca_df = pd.concat([pca_df, metadata[[batch_col]]], axis=1)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=batch_col, style=batch_col, s=100)
    plt.title(title)
    plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.legend()
    plt.savefig(save_file)
    plt.close()
