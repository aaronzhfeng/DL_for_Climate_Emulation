import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

BASE = 'best_unet'
ABLATIONS = {
    'unet_bilinear': 'unet_bilinear_vs_original.png',
    'unet_noskip': 'unet_noskip_vs_original.png',
}

def _load_metrics(model):
    path = os.path.join(RESULTS_DIR, model, 'metrics.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing')
    return pd.read_csv(path)

def _final_val_rmse(df):
    df = df[df['epoch'] != 'test'].copy()
    df['epoch'] = pd.to_numeric(df['epoch'])
    last = df.iloc[df['epoch'].idxmax()]
    return last['val/tas/rmse'], last['val/pr/rmse']

def _gather(model):
    df = _load_metrics(model)
    return dict(zip(['tas', 'pr'], _final_val_rmse(df)))

def main():
    try:
        base = _gather(BASE)
    except FileNotFoundError:
        print(f'metrics not found for {BASE}')
        return
    for ablation, fname in ABLATIONS.items():
        try:
            abl = _gather(ablation)
        except FileNotFoundError:
            print(f'metrics not found for {ablation}')
            continue
        dfp = pd.DataFrame({'Best U-Net': base, ablation: abl}).T
        x = np.arange(len(dfp))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(x - width/2, dfp['tas'], width, label='tas')
        ax.bar(x + width/2, dfp['pr'], width, label='pr')
        ax.set_xticks(x)
        ax.set_xticklabels(dfp.index, rotation=45, ha='right')
        ax.set_ylabel('Validation RMSE')
        ax.set_title(f'{ablation} vs Best U-Net')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)

if __name__ == '__main__':
    main()
