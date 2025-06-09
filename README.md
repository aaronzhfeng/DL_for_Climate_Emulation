
# DL_for_Climate_Emulation

**Climate Emulation with U-Net (CSE 151B, Spring 2025)**

> Predict monthly surface temperature (*tas*) and precipitation (*pr*)  
> from greenhouse gas forcings using deep learning models.

---

## 📁 Project Structure
```
DL_for_Climate_Emulation/
│
├── baseline.ipynb # Baseline models
├── Best-UNet.py # Main training script
├── DCNN.ipynb # Dilated CNN baseline
├── EDA.ipynb # Exploratory data analysis
├── FNO.ipynb # Fourier Neural Operator model
├── ResNet.ipynb # ResNet-based model
├── U-Net.ipynb # 2D U-Net baseline
├── U-Net-1D.ipynb # 1D U-Net variation
├── visualization.ipynb # Model output visualization
│
├── results/              # metrics for each trained model
│   └── <model_name>/metrics.csv
├── visualization/
│   ├── baseline_vs_unet.py          # Baseline models vs U-Net
│   ├── baseline_vs_fno_resnet.py    # Baselines vs FNO/ResNet
│   ├── unet_ablation.py             # Ablation comparison plots
│   └── figures/                     # Saved plots
│
├── environment.yaml # Conda environment specification
├── README.md # This file
│
├── processed_data_cse151b_v2_corrupted_ssp245/ # Preprocessed Zarr data (~9GB)
├── val_preds.npy # Saved validation predictions
├── val_trues.npy # Saved validation ground truth
├── eda_report/ # Output figures / stats from EDA
└── submissions/ # Sample prediction files
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
python visualization/baseline_vs_fno_resnet.py
python visualization/unet_ablation.py
```
Each script reads its needed `results/<model_name>/metrics.csv`
and saves figures under `visualization/figures/`.

---

### Additional Notebooks
EDA.ipynb, baseline.ipynb – Explore data and view simple baselines

ResNet.ipynb, U-Net-1D.ipynb, FNO.ipynb, DCNN.ipynb – Model comparisons

visualization.ipynb – Visualize predictions vs. ground truth

📌 Notes
Target resolution: 48×72 spatial grid

Input format: Zarr dataset

Key metrics: RMSE, Time-Mean RMSE, Time-Stddev MAE

📚 Citation
This project was developed as part of CSE 151B: Deep Learning
at UC San Diego, Spring 2025.
Please cite our final report if using this codebase.
