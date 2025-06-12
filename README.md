# Install necessary packages
```
python3 -m venv preprocessing # make a local python environment
source preprocessing/bin/activate
pip install --upgrade pip
pip install -r scripts/packages.txt
```
# Parameters 
```
Required parameters.
--data. Input dataframe, rows are samples and columns are features.
--metadata. A dataframe containing the batch information as a column
--output. Output file in tsv format. 
--max_missing_sample. The max allowed proportion of missing value in a sample. 
--log. Log2 or log10 transformation of data. pseudo count is 1, or specified with --pseudo_count.
--impute. Impute missing value with 'knn' (5 neighbour), 'min', or 'pimms'.
--batch_control. Specify the batch column in metadata, in order to remove batch effect using combat. 

Optional parameters. 
--plot. Plot the PCA and generate a PDF report. 
--pseudo_count. Pseudo count for log transformation. 
--normalize. Other than log, you can further normalize with 'quantile' or 'median'. 
--scale. Other than log, you can further scale with 'zscore' or 'pareto'. 
```

# Run the script
```
python scripts/preprocessing.py -h # check parameters to use
python scripts/preprocessing.py --data data_input/data.tsv --metadata data_input/metadata.tsv --output test.tsv --log log2 --impute knn --batch_control plate
```


