# When to use
This script is to preprocess proteomics or metabolomics raw data, where you can choose different methods to do quality control, missing value imputation, log transformation, batch effect removal, etc. 
Besides, the script also generates PCA plot for data produced in each step to help you follow how data is processed. 
# Install necessary packages
```
conda create -n ProteoPrep python=3.10 -y
conda activate ProteoPrep
pip install -r scripts/packages.txt
```
# Parameters

Required parameters:

- `--data`: Input dataframe, rows are samples and columns are features.
- `--metadata`: A dataframe containing the batch information as a column.
- `--output_dir`: Output directory where all results will be saved.  
- `--max_missing_sample`: Max allowed proportion of missing value in a sample. 
- `--log`: Log2 or log10 transformation of data. Pseudo count is 1, or set with `--pseudo_count`.
- `--impute`: Impute missing values with [knn](https://scikit-learn.org/stable/modules/impute.html), [pimms](https://github.com/RasmussenLab/pimms), or 'min'. 
- `--batch_control`: Specify the batch column in metadata to remove batch effect using [ComBat](https://github.com/epigenelabs/inmoose).

Optional parameters:

- `--disable_plot_PCA`: Disable PCA plot (default: show PCA).
- `--pseudo_count`: Pseudo count for log transformation.
- `--normalize`: Normalize with [quantile](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html) or `'median'`.
- `--scale`: Scale with `'zscore'` or `'pareto'`.
- `--save_intermediate`: Save intermediate results (especially for pimms) (default: False).

# Run the script

## Command line 
```
python scripts/preprocessing.py -h # check parameters to use
python scripts/preprocessing.py --data data_input/data.tsv --metadata data_input/metadata.tsv --output_dir output  --log log2 --impute knn --batch_control plate --max_missing_sample 0.4
```

## Run interactively with your browser 

```
streamlit run ./script/app.py # run it locally
nohup streamlit run ./scripts/app.py > streamlit.log 2>&1 & # or run in background. You can close the terminal or browser, typing the URL to your broswer will get you back to the interface. 
```


