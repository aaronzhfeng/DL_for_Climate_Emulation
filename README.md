
# DL_for_Climate_Emulation

**Climate Emulation with U-Net (CSE 151B, Spring 2025)**

> Predict monthly surface temperature (*tas*) and precipitation (*pr*)  
> from greenhouse gas forcings using deep learning models.

---

## 📁 Project Structure
```
DL_for_Climate_Emulation/
│
├── Best-UNet.py                      # Main training script for best U-Net
├── *.ipynb                           # Model training notebooks (baseline, UNet, FNO, etc.)
├── val_preds.npy                     # Saved validation predictions
├── val_trues.npy                     # Saved validation ground truth
│
├── eda_report/                       # Output figures / stats from EDA
├── processed_data_cse151b_v2_corrupted_ssp245/  # Preprocessed Zarr dataset
│
├── environment.yaml                  # Conda environment specification
├── README.md                         # Project documentation (this file)
│
├── models/
│   ├── ablation/
│   │   ├── unet_bilinear.py
│   │   └── unet_noskip.py
│   ├── baseline/
│   │   ├── cnn.py
│   │   ├── linear.py
│   │   └── mlp.py
│   ├── fno/
│   │   └── fno.py
│   ├── resnet/
│   │   └── resnet.py
│   └── unet/
│       ├── best_unet.py
│       └── metrics_logger.py
│
├── results/
│   ├── best_unet/
│   ├── cnn_baseline/
│   ├── fno/
│   ├── linear_baseline/
│   ├── mlp_baseline/
│   ├── resnet/
│   ├── unet_bilinear/
│   └── unet_noskip/
│
├── submissions/                      # Final model submission files
│
├── visualization/
│   ├── figures/                      # Saved output plots
│   ├── baseline_vs_unet.py
│   ├── unet_ablation.py
│   ├── unet_vs_fno_resnet.py
│   ├── baseline_vs_unet.ipynb
│   ├── best_unet_diagnostics.ipynb
│   ├── unet_ablation.ipynb
│   └── unet_vs_fno_resnet.ipynb

```
## ⚙️ Getting Started

### 1. Set Up the Environment

```
conda env create -f environment.yaml
conda activate climate-emulation
```


### 2. Prepare the Data
You can either:

Option A: 

Download and unzip the dataset into data/:
```
wget <download_link> -O data/processed_data_cse151b_v2_corrupted_ssp245.zarr.zip
unzip data/processed_data_cse151b_v2_corrupted_ssp245.zarr.zip
```


Option B: 

Use the existing directory:
processed_data_cse151b_v2_corrupted_ssp245/


### 3. Train and Evaluate the U-Net
```
python Best-UNet.py --config environment.yaml
```

### 4. Generate Comparison Figures
After training all models, run the visualization scripts:
```
python visualization/baseline_vs_unet.py
python visualization/unet_vs_fno_resnet.py
python visualization/unet_ablation.py
```
Each script reads its needed `results/<model_name>/metrics.csv`
and saves figures under `visualization/figures/`.

---

### Additional Notebooks
EDA.ipynb, baseline.ipynb – Explore data and view simple baselines

ResNet.ipynb, U-Net-1D.ipynb, FNO.ipynb, DCNN.ipynb – Model comparisons

visualization/ - figures and their ipynb/py files for the generation

📌 Notes
Target resolution: 48×72 spatial grid

Input format: Zarr dataset

Key metrics: RMSE, Time-Mean RMSE, Time-Stddev MAE

📚 Citation
This project was developed as part of CSE 151B: Deep Learning
at UC San Diego, Spring 2025.
Please cite our final report if using this codebase.
