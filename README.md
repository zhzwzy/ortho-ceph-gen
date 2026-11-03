# Source Code

**Associated manuscript:** Developing a systemic interactive thinking-based framework bridging high-dimensional complexity and clinical interpretability for orthodontic treatment planning  
**Manuscript ID:** NCOMMS-26-090224

## Contents

- `code/diffusion/`: cephalometric image and landmark processing; ControlNet and UNet LoRA training.
- `code/metric_pred/`: landmark measurement, dataset construction, metric prediction, and result visualization.

## Software environment

Python 3.9.23

- Diffusion: PyTorch 2.8.0+cu126, Diffusers 0.32.2, Transformers 4.49.0, PEFT 0.17.1, NumPy 2.0.1, pandas 2.3.3, OpenCV 4.12.0.88, Matplotlib 3.9.4, and xFormers 0.0.32.post2 (optional).
- Metric prediction: scikit-learn 1.6.1, seaborn 0.13.2, joblib 1.5.3, CatBoost 1.2.10, LightGBM 4.6.0, XGBoost 2.1.4, openpyxl 3.1.5, and Pillow 11.0.0; NumPy, pandas, and Matplotlib versions are listed above.

The VLM module is not publicly released at this time. This package contains source code only; datasets, checkpoints, model weights, and results are excluded.
