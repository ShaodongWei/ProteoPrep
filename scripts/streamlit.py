import streamlit as st
import pandas as pd
import subprocess
import tempfile
import os
import zipfile
import io

st.set_page_config(page_title="ProteoPrep Streamlit", layout="wide")
st.title("🧪 ProteoPrep - Web Preprocessing Interface")

# =========================
# Upload input files
# =========================
st.subheader("📤 Upload Input Files")
input_data = st.file_uploader("Upload proteomics data (CSV/TSV)", type=["csv", "tsv"])

# =========================
# Parameter configuration
# =========================
st.subheader("⚙️ Configure Parameters")

max_missing = st.slider("Maximum missing values per sample", 0.0, 1.0, 0.4)
log_method = st.selectbox("Log Transformation", ["log2", "log10"])
impute_method = st.selectbox("Imputation Method", ["pimms_vae", "pimms_dae", "pimms_cft", "knn"])
normalize_method = st.selectbox("Normalization Method", ["median", "quantile", "None"])
pseudo_count = st.number_input("Pseudo count for log transform", value=1.0)

plot_pca = st.checkbox("Save PCA plots", value=True)

do_batch = st.checkbox("Apply batch correction", value=False)
batch_col = st.text_input("Batch control column name", value="", disabled=not do_batch)

# metadata is only needed if PCA plots or batch correction is enabled
need_metadata = plot_pca or do_batch

input_meta = st.file_uploader(
    "Upload metadata file (CSV/TSV) (required if PCA plots or batch correction enabled)",
    type=["csv", "tsv"],
    disabled=not need_metadata
)

# =========================
# Run pipeline
# =========================
st.subheader("🚀 Run")

run_clicked = st.button("Run Preprocessing")

if run_clicked:
    # Basic checks
    if input_data is None:
        st.error("Please upload a data file.")
        st.stop()

    if need_metadata and input_meta is None:
        st.error("Metadata is required when PCA plots or batch correction is enabled.")
        st.stop()

    if do_batch and batch_col.strip() == "":
        st.error("Batch control column name cannot be empty when batch correction is enabled.")
        st.stop()

    with st.spinner("Processing..."):

        # Save input data to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv") as f1:
            f1.write(input_data.read())
            data_path = f1.name

        # Save metadata only if provided/required
        meta_path = None
        if input_meta is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv") as f2:
                f2.write(input_meta.read())
                meta_path = f2.name

        output_dir = tempfile.mkdtemp()

        # Build command (IMPORTANT: preprocessing.py must accept optional --metadata)
        cmd = [
            "python", "./scripts/preprocessing.py",
            "--data", data_path,
            "--output_dir", output_dir,
            "--max_missing_sample", str(max_missing),
            "--log", log_method,
            "--impute", impute_method,
            "--pseudo_count", str(pseudo_count),
        ]

        if meta_path is not None:
            cmd += ["--metadata", meta_path]

        if normalize_method != "None":
            cmd += ["--normalize", normalize_method]

        if do_batch:
            cmd += ["--batch_control", batch_col.strip()]

        # Your CLI flag is inverted: if user does NOT want PCA, pass --disable_plot_PCA
        if not plot_pca:
            cmd += ["--disable_plot_PCA"]

        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            st.error("❌ Pipeline error")
            if result.stderr:
                st.code(result.stderr)
            if result.stdout:
                st.code(result.stdout)
            st.stop()

        # Load and show output
        output_file = os.path.join(output_dir, "cleaned_data.tsv")
        if not os.path.exists(output_file):
            st.error("Output file not found. Check script output below:")
            st.code(result.stdout)
            st.stop()

        df_out = pd.read_csv(output_file, sep="\t")
        st.success("✅ Preprocessing completed!")
        st.dataframe(df_out)

        # Download cleaned_data.tsv
        st.download_button(
            label="📥 Download Cleaned Data (TSV)",
            data=df_out.to_csv(sep="\t", index=False).encode("utf-8"),
            file_name="cleaned_data.tsv",
            mime="text/tsv",
        )

        # Zip the whole output_dir
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path):
                    zip_file.write(file_path, arcname=filename)
        zip_buffer.seek(0)

        st.download_button(
            label="📦 Download All Outputs as ZIP",
            data=zip_buffer,
            file_name="proteoprep_outputs.zip",
            mime="application/zip",
        )
