import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = [
    'linear_baseline',
    'mlp_baseline',
    'cnn_baseline',
    'best_unet',
]

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

def main():
    metrics = {}
    for m in MODELS:
        try:
            df = _load_metrics(m)
        except FileNotFoundError:
            print(f'metrics not found for {m}')
            continue
        metrics[m] = dict(zip(['tas', 'pr'], _final_val_rmse(df)))
    if not metrics:
        print('no metrics found')
        return
    dfp = pd.DataFrame(metrics).T
    x = np.arange(len(dfp))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x - width/2, dfp['tas'], width, label='tas')
    ax.bar(x + width/2, dfp['pr'], width, label='pr')
    ax.set_xticks(x)
    ax.set_xticklabels(dfp.index, rotation=45, ha='right')
    ax.set_ylabel('Validation RMSE')
    ax.set_title('Baselines vs U-Net')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'baseline_vs_unet.png'), dpi=300)

if __name__ == '__main__':
    main()
