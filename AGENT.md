# CSE151B Final Project – Visualization Plan

In order to effectively communicate our results and insights, we will prepare a comprehensive set of figures covering model performance, ablation studies, training dynamics, model architecture, and qualitative analyses. All plots will use **RMSE** as the evaluation metric (for both temperature **tas** and precipitation **pr**), and each figure will be self-contained with clear axes, legends, and descriptive captions. We will ensure a consistent, **NeurIPS-style** presentation – meaning figures are neat, legible, and use sufficiently large fonts and line weights for clarity. Below is a detailed guide to each required visualization:

## Baseline Performance Comparison (U-Net vs. CNN, MLP, Linear)

**Content:** The first figure will compare the predictive performance of our **U-Net** model against baseline models (a simple CNN, a 2-layer MLP, and a linear regression) using RMSE. This highlights how much U-Net improves over simpler architectures. We will show **validation RMSE** for each model on the two target variables (**tas** and **pr**). For example, in our results the U-Net’s RMSE on tas is dramatically lower (\~4.15) than the linear model’s (\~13.01), and similarly for pr (U-Net \~2.83 vs. linear \~3.57). This confirms the **significant performance gain** achieved by the U-Net over both linear and shallow nonlinear baselines.

**Design:** A **bar chart** is ideal for this comparison, making differences immediately visible. We can use a grouped-bar format with each model as a group and two bars within each group (one for **tas RMSE**, one for **pr RMSE**). The x-axis will list the models (Linear, MLP, CNN, U-Net), and the y-axis will be RMSE (with a suitable range to accommodate the highest error). Each bar should be labeled with its exact RMSE value for clarity. Alternatively, if separating the variables makes the chart too crowded, we could use two side-by-side plots – one for tas and one for pr – each showing the RMSE of all models. We’ll use distinct colors or hatching for each model (or each variable) and include a legend if needed (e.g. color-coded for tas vs pr). The **caption** will clearly state that we are comparing validation RMSE of the four models (lower is better) for each target variable. By examining this plot, readers should quickly grasp that the U-Net outperforms the baseline models by a wide margin in terms of RMSE.

## U-Net Ablation Study (Skip Connections & Upsampling)

**Content:** Next, we will include an **ablation study plot** focusing on the U-Net architecture variants. We vary two key components of our U-Net: (1) **Skip connections** (enabled vs. disabled) and (2) **Upsampling method** (learned transposed convolution vs. fixed bilinear interpolation). This experiment will result in **four U-Net variants**: with skip vs. without skip, each using either bilinear upsampling or transposed conv. The goal is to illustrate how much each component contributes to performance. We expect, based on theory and our results, that removing skip connections will significantly degrade performance (since skip connections help preserve spatial detail), and the upsampling method may also affect accuracy (transposed convolutions can learn to upsample sharper details, whereas bilinear might be smoother). By comparing RMSE across these variants, we can confirm which design choices were most impactful.

**Design:** We will use a **grouped bar chart** for this ablation as well. For clarity, the x-axis can be divided into two groups: **“With Skip Connections”** and **“No Skip Connections.”** Within each group, include two bars representing the upsampling method (e.g. **Bilinear vs. Transposed Conv**). This layout lets readers directly see the effect of removing skip connections (by comparing between groups) as well as the effect of changing the upsampling (by comparing bars within each group). The y-axis will be **Validation RMSE** (we could plot separate bars for tas and pr here as well, but it may be clearer to focus on a single aggregated metric per variant – for instance, we might plot the RMSE for **tas** only, or present two subplots for tas and pr if space allows). Each bar will again have its exact RMSE annotated on top for precision. The **caption** will describe the four U-Net variants and note observations, e.g. “U-Net without skip connections performs substantially worse, confirming the importance of skip connections in preserving spatial features. The model with transposed convolution slightly outperforms the bilinear upsampling variant, suggesting learned upsampling gives a minor advantage (as seen by lower RMSE).” This plot directly addresses the ablation requirements by quantifying the contribution of these two architectural components.

## Training vs. Validation Loss Curve

**Content:** To give insight into the model training process and potential overfitting, we will include a **training and validation loss curve** for the U-Net. This plot shows how the loss (error) on the training set and validation set evolves over epochs. It helps demonstrate that our training was well-behaved and where training may have stopped. In our case, we observed that the U-Net’s loss steadily decreased and then **plateaued around epoch 12**, with very little gap between training and validation curves (indicating minimal overfitting). Including this figure is optional but helpful to convince the reader that the training converged properly and early stopping was used.

**Design:** The training plot will be a **line graph** with epochs on the x-axis and loss on the y-axis. We can use **RMSE** on the y-axis to be consistent with evaluation metric (even if the model optimized MSE internally, plotting RMSE = √MSE makes it more interpretable). There will be two lines – **training RMSE** (e.g. in blue, perhaps dashed line) and **validation RMSE** (in red, solid line). We will clearly annotate or legend these lines. If early stopping was employed, we can mark the epoch at which training stopped or the best model epoch (for example, a vertical line at epoch 12 or a star marker on the curves). The axes will be labeled (e.g. “Epoch” and “RMSE (training vs validation)”), and the y-scale can start at 0 and go up to the initial loss value. The **caption** will note key aspects, for instance: “Training and validation RMSE over epochs for the U-Net model. The validation curve closely tracks the training curve and plateaus after \~12 epochs, indicating good convergence without overfitting. Early stopping halted training at the point of minimal validation loss.” This figure gives the reader confidence that the model training was sufficiently long and appropriately regularized.

*(If space is tight in the report, this figure could be moved to an appendix or slightly reduced, but including it in the main paper is valuable for a **NeurIPS-style** report since it demonstrates the training dynamics.)*

## U-Net Architecture Schematic

&#x20;**Figure:** **U-Net model architecture** used in our project, shown in the classic “U” configuration of a **contracting encoder** (left) and **expanding decoder** (right). The network consists of four downsampling steps (blue boxes on the left, each reducing spatial dimensions and increasing channel depth) and four matching upsampling steps (blue boxes on the right). At each decoder step, the feature map is **upsampled** (green upward arrows) and concatenated with the corresponding encoder feature map via a **skip connection** (horizontal gray arrows), which restores high-resolution details to the decoder. The numbers atop the boxes indicate the number of feature channels, and the vertical size of each box suggests the relative spatial resolution. Our implementation closely follows this design: an input (e.g. climate data map) is processed down to a low-resolution latent representation (bottom of the “U”), then rebuilt to original resolution in the decoder, yielding the final prediction map. *Note:* In our ablations we experiment with removing the skip connections (which would effectively turn this into a plain autoencoder structure, missing the lateral arrows) and with two upsampling methods – the diagram shows transposed convolution layers as green arrows, but we also try a bilinear interpolation + convolution approach (not explicitly shown). This schematic would be placed in the **Methods/Architecture** section of the report to help readers visualize the model structure before seeing the results.

## Qualitative Results: Ground Truth vs. Prediction Maps

For a more intuitive evaluation of model performance, we will include **qualitative visualization of predictions** versus the ground truth. Specifically, we will show example **maps of the climate variables**: surface air temperature (**tas**) and precipitation rate (**pr**), comparing the model’s prediction to the true values for the **same sample** (at a given time step). This allows us to see how well the U-Net captures the spatial patterns, magnitudes, and extreme values in the data.

**Design:** We will create a figure with a **side-by-side layout** for easy comparison. One effective layout is a **2×2 grid** of subplots:

* **(a) Ground Truth tas** – a map of the true temperature field.
* **(b) Predicted tas** – the U-Net’s predicted temperature field for that time.
* **(c) Ground Truth pr** – a map of the true precipitation field.
* **(d) Predicted pr** – the U-Net’s predicted precipitation field.

Each map will be plotted with a color scale (e.g. a heatmap or geographic contour plot) appropriate to the variable’s range. We will use the **same color scale and range for each pair** (ground truth vs prediction) so that differences are visually apparent. For instance, if tas ranges from, say, 250–310 K in the data, we’ll use a consistent colorbar for both ground truth and predicted tas plots. We’ll do the same for pr (which might have a highly skewed distribution, so we might use, for example, a logarithmic color scale or a “rainfall” color palette to highlight differences). Axes will be labeled with latitude and longitude (if applicable), or with index coordinates if these are small grid patches, and each subplot will have a title above it indicating what is shown (“True tas”, “Predicted tas”, etc.).

The **caption** for this figure will describe that it’s illustrating a representative example, and highlight the qualitative performance: e.g. *“Comparison of **ground truth** and **U-Net predicted** maps for **tas** (top row) and **pr** (bottom row) for a particular time step. The U-Net successfully reproduces the broad spatial patterns in both temperature and precipitation. Tas predictions closely match true values except for slight smoothing of fine-scale details, while pr predictions capture regional rainfall patterns but tend to underestimate some localized heavy precipitation spots.”* This narrative in the caption helps the reader understand strengths and weaknesses observed in the visual comparison.

## Error Analysis: Worst-Performing Examples

To complement the average performance metrics, we will analyze cases where the model performed **poorly (highest error cases)**. Specifically, we will identify a few **worst-case samples** (e.g. the top 2–3 samples with highest RMSE error) and visualize them to understand failure modes. Often in climate data, the largest errors might correspond to extreme events or out-of-distribution scenarios.

**Design:** We can present these worst cases in a figure similar to the above, but focusing on the error. For each selected sample, we will show a **difference map** in addition to the usual prediction vs truth, to pinpoint where errors are large. For example, for an extreme precipitation event (which our milestone results indicated was a common failure case), we could show:

* Ground truth pr vs predicted pr, along with a **prediction error map** (prediction minus truth) for pr.

If multiple samples are shown, we might arrange them in a multi-panel figure (each sample being one column, for instance). **Colorbars** on difference maps will be centered at zero (diverging colormap) so we can see positive vs negative errors. In one of our worst cases, we found that errors concentrated in **equatorial regions during extreme precipitation events**, where the model **underestimated peak pr values**. A difference plot for such a sample would clearly show a large negative error at the location of the heaviest rainfall (the model under-predicted the magnitude). We will annotate any notable regions or values, if necessary (e.g. circle the area of an extreme event).

The **caption** will explain what each panel represents and draw attention to the pattern of errors. For example: *“Visualization of one of the **worst-performing examples** for precipitation. The model fails to predict an extreme rainfall event in the equatorial Pacific, as seen by the large discrepancy between true and predicted pr (panel c vs d) and the strong negative errors in the difference map (panel e, blue region) indicating underestimation of precipitation intensity. Such extreme events are rare in training, making them challenging for the model to learn.”* By analyzing these cases visually, we set the stage for discussing **why** the model might be failing here.

## Failure Patterns and Proposed Improvements

Finally, we will include a concise **discussion** (textual, possibly alongside the error analysis figure or in a separate subsection) of the **common failure patterns** observed and **suggestions for improvement**. Based on the error analysis and validation results, a few key issues have emerged:

* **Extreme Value Underestimation:** The U-Net tends to smooth out or underestimate extremely high values (especially for precipitation). This is evident in the worst-case examples where peak rainfall is under-predicted. This could be due to the loss function (MSE/RMSE) equally weighing all errors, thus not focusing enough on rare extremes. **Improvement:** We can address this by tweaking the loss function – for instance, using a **weighted loss** or a **Hybrid loss** that gives extra weight to high-error regions or using a robust loss (like Huber) that might handle outliers better. Another approach is to perform data augmentation or oversampling for extreme-event examples so that the model gets more practice on these cases. We might also consider modeling extreme events separately or adding an auxiliary objective focused on peaks.

* **Label Noise or Data Issues:** We suspect that some high errors (notably for tas in the test set) might be due to **inconsistent or corrupted labels** rather than model error. If certain ground truth values are erroneous (e.g. a glitch in the dataset reporting absurd temperatures), the model’s predictions will look wrong by comparison even if the model was actually reasonable. **Improvement:** In a real-world setting, one should **verify data quality**. If label noise is present, possible remedies include cleaning the dataset (if feasible), or using loss functions that are less sensitive to outliers (which overlaps with the robust loss suggestion above). In our report, we noted the plan to manually inspect such cases and perhaps remove or correct faulty data. Communicating to the reader that some errors might not be the model’s fault is also important.

* **Out-of-Distribution Inputs / Spatial Bias:** Some errors might occur in particular **regions or regimes** that were under-represented in training. For example, if the model consistently has higher RMSE in polar regions or desert areas, this could indicate a **spatial bias** or an **OOD (out-of-distribution) input** issue. Perhaps the climate in those areas wasn’t well covered by training data, or the model lacks mechanisms to handle those differences. **Improvement:** We can suggest adding **input features or coordinates** to help the model learn regional differences, or incorporating techniques like **spatial attention** or a specialized architecture (e.g. using a Fourier Neural Operator layer or spherical convolution for global data) to better capture global patterns. Additionally, gathering more training data covering those regimes or using transfer learning from a model pretrained on similar data could help.

* **Resolution and Fine-Scale Detail:** By design, the U-Net should capture fine-scale detail thanks to skip connections. However, if we noticed any consistent blurring of small features (perhaps in tas field if sharp fronts or gradients are smoothed), it indicates the model’s resolution limit. **Improvement:** Increasing the **model capacity** (more filters or an extra level in the U-Net), or using **progressive upsampling** could improve fine-scale accuracy. We might also experiment with **post-processing** techniques to sharpen the output (though that’s outside pure ML, it can be mentioned as a potential step).

Each of these failure modes and suggestions will be described in the text accompanying the error analysis. This discussion is crucial in a NeurIPS-style report to demonstrate understanding of the results and to propose future work. We will ensure the text references the figures as needed (e.g. “as seen in Figure X, the model underestimates extreme precipitation”) and is written clearly. By providing **concrete suggestions** (like loss function adjustments, data augmentation, and architectural tweaks), we address the rubric’s requirement for proposing improvements, showing a thoughtful reflection on how to tackle the model’s limitations.

---

**In summary**, the final report will contain a rich set of visuals: a **baseline comparison** figure highlighting U-Net’s performance gains, an **ablation study** figure demonstrating the importance of skip connections and upsampling choice, a **training curve** to show learning dynamics, a **model architecture diagram** for clarity, and **qualitative maps** (both typical cases and failure cases) for deeper insight. All figures will be designed and presented in a polished, **publication-quality manner** – with readable axes, consistent styling, and informative captions – meeting the standards of a NeurIPS-style paper. This cohesive collection of figures will not only satisfy the assignment requirements but also greatly aid in conveying our project’s findings and the reasoning behind them.



**Implementation Plan**


1. **Create model scripts**  
   Extract code from notebooks and place them into modular Python files under `models/`. Each script should accept arguments for data path, hyperparameters, and output directory. Leverage the existing `ClimateDataModule` and `ClimateEmulationModule` classes from `Best-UNet.py`.

2. **Standardize metric logging**  
   Write a common callback (`MetricsLogger`) or utility function that every model script uses. Ensure each training run saves epoch-level metrics and final RMSE values to CSV.

3. **Train and save results**  
   Run each script (U-Net, baselines, ablation variants) to produce their `metrics.csv` and predictions. These CSV files will be the data source for the visualizations.

4. **Implement visualization script**  
   `visualization/plot_results.py` reads all CSVs and produced predictions to generate:
   - Baseline comparison bar plot.
   - Ablation study bar plot.
   - Training vs validation curve (from U-Net metrics).
   - Qualitative maps and worst-case error plots (using the saved predictions/targets).  
     Save all figures to `visualization/figures/`.

5. **Update README**  
   Document how to run each model script, where metrics are saved, and how to run the visualization script to reproduce the figures. Show the updated repository structure in the README.

6. **(Optional) Generate an architecture diagram**  
   Use a tool like `graphviz` or `torchviz` to create a U-Net schematic and save it under `visualization/figures/`.


**Proposed Directory Layout**
DL_for_Climate_Emulation/
├── models/
│   ├── baseline/
│   │   ├── linear.py
│   │   ├── mlp.py
│   │   └── cnn.py
│   ├── unet/
│   │   ├── best_unet.py  (existing Best-UNet.py)
│   │   └── unet_variant.py  (skip/bilinear options)
│   ├── resnet/
│   │   └── resnet.py
│   ├── fno/
│   │   └── fno.py
│   └── ablation/
│       ├── unet_noskip.py
│       └── unet_bilinear.py
├── notebooks/    # original .ipynb files
├── results/
│   └── <model_name>/metrics.csv
├── visualization/
│   ├── plot_results.py
│   └── figures/
└── ...
