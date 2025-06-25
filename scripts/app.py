import streamlit as st
import pandas as pd
import subprocess
import tempfile
import os
import zipfile
import io

st.set_page_config(page_title="ProteoPrep Streamlit", layout="wide")
st.title("🧪 ProteoPrep - Web Preprocessing Interface")

# Upload input files
st.subheader("📤 Upload Input Files")
input_data = st.file_uploader("Upload proteomics data (CSV/TSV)", type=["csv", "tsv"])
input_meta = st.file_uploader("Upload metadata file (CSV/TSV)", type=["csv", "tsv"])

# Parameter configuration
st.subheader("⚙️ Configure Parameters")
max_missing = st.slider("Maximum missing values per sample", 0.0, 1.0, 0.4)
log_method = st.selectbox("Log Transformation", ["log2", "log10"])
impute_method = st.selectbox("Imputation Method", ["pimms", "knn", "min"])
batch_col = st.text_input("Batch control column name", value="plate")
pseudo_count = st.number_input("Pseudo count for log transform", value=1.0)
normalize_method = st.selectbox("Normalization Method (optional)", [None, "median", "quantile"])
scale_method = st.selectbox("Scaling Method (optional)", [None, "zscore", "pareto"])
save_intermediate = st.checkbox("Save intermediate results", value=False)
plot_pca = st.checkbox("Save PCA plots", value=True)

if input_data and input_meta:
    if st.button("🚀 Run Preprocessing"):
        with st.spinner("Processing..."):

            # Save input files to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f1:
                f1.write(input_data.read())
                data_path = f1.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f2:
                f2.write(input_meta.read())
                meta_path = f2.name

            output_dir = tempfile.mkdtemp()

            # Build command
            cmd = [
                "python", "./scripts/preprocessing.py",
                "--data", data_path,
                "--metadata", meta_path,
                "--output_dir", output_dir,
                "--max_missing_sample", str(max_missing),
                "--log", log_method,
                "--impute", impute_method,
                "--batch_control", batch_col,
                "--pseudo_count", str(pseudo_count)
            ]
            if normalize_method:
                cmd += ["--normalize", normalize_method]
            if scale_method:
                cmd += ["--scale", scale_method]
            if save_intermediate:
                cmd += ["--save_intermediate"]
            if not plot_pca:
                cmd += ["--disable_plot_PCA"]

            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                st.error("❌ Pipeline error")
                st.code(result.stderr)
            else:
                output_file = os.path.join(output_dir, "cleaned_data.tsv")
                if os.path.exists(output_file):
                    df_out = pd.read_csv(output_file, sep="\t")
                    st.success("✅ Preprocessing completed!")
                    st.dataframe(df_out)

                    # Download only cleaned_data.tsv
                    st.download_button(
                        label="📥 Download Cleaned Data (TSV)",
                        data=df_out.to_csv(sep="\t", index=False).encode("utf-8"),
                        file_name="cleaned_data.tsv",
                        mime="text/tsv"
                    )

                    # Create ZIP archive of the whole output_dir
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for filename in os.listdir(output_dir):
                            file_path = os.path.join(output_dir, filename)
                            zip_file.write(file_path, arcname=filename)
                    zip_buffer.seek(0)

                    st.download_button(
                        label="📦 Download All Outputs as ZIP",
                        data=zip_buffer,
                        file_name="proteoprep_outputs.zip",
                        mime="application/zip"
                    )

                else:
                    st.error("Output file not found. Check script output below:")
                    st.code(result.stdout)
