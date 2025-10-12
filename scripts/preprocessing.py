# import functions 
from functions import (
    load_data,
    remove_low_quality,
    remove_outlier_sample,
    impute_missing,
    log_transform,
    remove_batch_effect,
    plot_pca
)

import argparse
import pandas as pd
import os 

# === Main Pipeline ===
def preprocessing():
    parser = argparse.ArgumentParser(description="Preprocess and QC proteomics/metabolomics data", allow_abbrev=False)
     
    # required arguments
    parser.add_argument('--data', required=True, help='Input data')
    parser.add_argument('--metadata', required=True, help='Metadata')
    parser.add_argument('--output_dir', required=True, help='Output directory to save cleaned data')
    parser.add_argument('--max_missing_sample', default=0.4, type=float, help='Maximum fraction of missing values allowed per sample (default 0.4)')
    parser.add_argument('--log', required=True, choices=['log2', 'log10'], help='log transformation ')
    parser.add_argument('--impute', required=True, choices=['pimms_vae','pimms_dae','pimms_cft','knn'], help='Imputation method. Options: pimms (VAE, DAE, CollaborativeFilteringTransformer) or knn (10 neighbors)')
    parser.add_argument('--batch_control', required=True, help='Batch column name in metadata')
    
    # optional arguments
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate results (default False)')
    parser.add_argument('--disable_plot_PCA', action='store_false', help='Disable PCA plot (default show PCA plot).')
    parser.add_argument('--pseudo_count', type=float, default=1, help='Pseudo count for log transformation (default 1)')
    
    args = parser.parse_args()

    # Load data and metadata
    os.listdir(os.getcwd())

    data, metadata = load_data(args.data, args.metadata)
    
    # Check if data and metadata are loaded correctly
    if data.empty or metadata.empty:
        raise ValueError("Data or metadata is empty. Please check the input files.")
    if data.shape[0] != metadata.shape[0]:
        raise ValueError("Number of samples in data and metadata do not match. Please check the input files.")
    if args.batch_control not in metadata.columns:
        raise ValueError(f"Batch control column '{args.batch_control}' not found in metadata. Please check the input files.")
    if args.log not in ['log2', 'log10']:
        raise ValueError(f"Log transformation method '{args.log}' is not supported. Choose 'log2' or 'log10'.")
    if args.impute not in ['pimms_vae','pimms_dae','pimms_cft', 'knn']:
        raise ValueError(f"Imputation method '{args.impute}' is not supported. Choose 'pimms_vae','pimms_dae','pimms_cft', 'knn'")
    
    # Check if output directory exists, if not create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Remove low quality data based on missing values
    data = remove_low_quality(data, missing_sample_thresh=args.max_missing_sample, missing_feature_thresh=1)
    data = remove_outlier_sample(data, threshold=1.5)
    if args.disable_plot_PCA:
        plot_pca(data, metadata, batch_col='plate', save_file=os.path.join(args.output_dir, 'pca_QC.pdf'), title='PCA after removing low quality data')

    # required preprocessing steps
    # log transform data
    data = log_transform(data, pseudo_count=args.pseudo_count, method=args.log)
    if args.disable_plot_PCA:
        plot_pca(data, metadata, batch_col='plate', save_file=os.path.join(args.output_dir, 'pca_log.pdf'), title='PCA after log transformation')

    # impute missing values
    data = impute_missing(data, method=args.impute, save_intermediate=args.save_intermediate)
    if args.disable_plot_PCA:
        plot_pca(data, metadata, batch_col='plate', save_file=os.path.join(args.output_dir, 'pca_missing_imputation.pdf'), title='PCA after missing value imputation')

    # Remove batch effects
    data = remove_batch_effect(data, metadata, batch_col=args.batch_control)
    if args.disable_plot_PCA:
        plot_pca(data, metadata, batch_col='plate', save_file=os.path.join(args.output_dir, 'pca_batch_correction.pdf'), title='PCA after batch correction')

    # Save the cleaned data
    output_file = os.path.join(args.output_dir, 'cleaned_data.tsv')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(output_file):
        print(f"Warning: Output file {output_file} already exists. It will be overwritten.")
    data = data.round(3)
    data = data.reset_index(drop=True)
    data.to_csv(output_file, sep='\t', index=False)

    print("Preprocessing complete. Cleaned data saved to:", output_file)

if __name__ == '__main__':
    preprocessing()
