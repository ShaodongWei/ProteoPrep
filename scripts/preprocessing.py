# import functions 
from functions import (
    load_data,
    remove_low_quality,
    impute_missing,
    log_transform,
    remove_batch_effect,
    normalize,
    scale
)

import argparse
import pandas as pd

# === Main Pipeline ===
def preprocessing():
    parser = argparse.ArgumentParser(description="Preprocess and QC proteomics/metabolomics data")
    
    # required arguments
    parser.add_argument('--data', required=True, help='Input data')
    parser.add_argument('--metadata', required=True, help='Metadata')
    parser.add_argument('--output', required=True, help='File path to save cleaned data')
    parser.add_argument('--max_missing_sample', default=0.4, type=float, help='Maximum fraction of missing values allowed per sample (default 0.4)')
    parser.add_argument('--log', required=True, choices=['log2', 'log10'], help='log transformation ')
    parser.add_argument('--impute', required=True, choices=['pimms','knn', 'min'], help='Imputation method')
    parser.add_argument('--batch_control', required=True, help='Batch column name in metadata')
    
    # optional arguments
    parser.add_argument('--pseudo_count', type=float, default=1, help='Pseudo count for log transformation (default 1)')
    parser.add_argument('--normalize', choices=['median', 'quantile'], help='Normalization method')
    parser.add_argument('--scale', choices=['zscore', 'pareto'], help='Scaling method')
    args = parser.parse_args()

    data, metadata = load_data(args.data, args.metadata)
    
    # Remove low quality data based on missing values
    data = remove_low_quality(data, missing_sample_thresh=args.max_missing_sample, missing_feature_thresh=1)

    # required preprocessing steps
    # log transform data
    data = log_transform(data, pseudo_count=args.pseudo_count, method=args.log)

    # impute missing values
    data = impute_missing(data, method=args.impute)
    
    # Remove batch effects
    data = remove_batch_effect(data, metadata, batch_col=args.batch_control)

    ## optional preprocessing steps
    # Normalize and scale data
    if args.normalize:
        data = normalize(data, method=args.normalize)

    # Scale data
    if args.scale:
        data = scale(data, method=args.scale)

    # Save the cleaned data
    # data.index = metadata.iloc[data.index]['sample'].values  # Set index to sample names
    # data.index.name = 'sample'  # Name the index
    data = data.round(3)  # Round data to 3 decimal places
    data.to_csv(args.output, sep='\t', index=True)

    print("Preprocessing complete. Cleaned data saved to:", args.output)

if __name__ == '__main__':
    preprocessing()
