
# DL_for_Climate_Emulation

**Climate Emulation with U-Net (CSE 151B, Spring 2025)**

> Predict monthly surface temperature (*tas*) and precipitation (*pr*)  
> from greenhouse gas forcings using deep learning models.

---

## 📁 Project Structure
```
DL_for_Climate_Emulation/
│
├── 151b_kaggle.ipynb # Kaggle submission template
├── Best-UNet.py # Main training script
├── DCNN.ipynb # Dilated CNN baseline
├── EDA.ipynb # Exploratory data analysis
├── FNO.ipynb # Fourier Neural Operator model
├── ResNet.ipynb # ResNet-based model
├── U-Net.ipynb # 2D U-Net baseline
├── U-Net-1D.ipynb # 1D U-Net variation
├── visualization.ipynb # Model output visualization
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

---

### Additional Notebooks
EDA.ipynb – Explore data and view simple baselines

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
