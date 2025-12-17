# DBR: Depth Barrier Regularization for Safe Vision Based Navigation

### TL;DR: DBR leverages monocular depth during training to inject a differentiable geometric prior into vision based navigation policies. It shapes the learned policy to avoid collisions without requiring depth or planning at test time which significantly improves safety metrics while maintaining navigation performance.

[![Paper](https://img.shields.io/badge/Paper-Google%20Drive-orange)](https://drive.google.com/file/d/1xFJzhZIBCb5IGV3D6hgymHjlrfn3jJHd/view?usp=sharing) [![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/harsh-sutariya/DBR)

DBR is built upon the [CityWalker](https://github.com/ai4ce/CityWalker) framework. While CityWalker learns robust navigation from web scale videos, it can struggle with near field obstacles and tight spaces. DBR addresses this by using monocular depth estimation (e.g., Depth Anything V2) to add a **differentiable barrier loss** that penalizes waypoint predictions violating a safety margin.

---

## Key Features

- **Train Time Only Depth**: DBR uses depth maps only during training to shape the policy. At inference, the model remains **RGB only**, maintaining computational efficiency.
- **Polar Clearance Representation**: Converts 2D depth maps into a 1D vector of yaw indexed clearances, aligning the safety signal with the robot's action space.
- **Differentiable Barrier Loss**: Uses a softplus based barrier function to enforce binary safety constraints (safe vs. unsafe) without over penalizing safe actions.
- **Model Agnostic**: DBR can be integrated into any waypoint based navigation architecture.

---

## Quantitative Results

DBR demonstrates significant safety improvements on the CityWalk dataset:

| Model | DVR $\downarrow$ | MDM $\uparrow$ | Arr. Acc. $\uparrow$ | AOE $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | 13.0% | 1.85m | **80.7%** | **8.3°** |
| **+DBR** | **11.0%** | **2.28m** | 80.5% | 8.5° |

- **15% reduction** in Depth Violation Rate (DVR).
- **23% increase** in Min-Depth Margin (MDM).
- Negligible impact on navigation accuracy.

---

## Getting Started

### Installation
The project is tested with Python 3.11, PyTorch 2.5.0, and CUDA 12.1.
```bash
conda env create -f environment.yml
conda activate citywalker
```

### Training with DBR
To train a model with Depth Barrier Regularization:
```bash
python train.py --config config/frodobots_dbr.yaml
```
Key hyperparameters in the config:
- `dbr.tau`: Safety margin (default: 0.5m)
- `dbr.lambda_bar`: Barrier loss weight (default: 1.0)
- `dbr.kappa`: Soft-min temperature (default: 20.0)

### Evaluation
Evaluate the model using safety metrics (DVR, MDM) alongside standard navigation metrics:
```bash
python test.py --config config/frodobots_dbr.yaml --checkpoint <path_to_checkpoint>
```

---

## Acknowledgements
We thank the [CityWalker](https://github.com/ai4ce/CityWalker) team for the base framework and datasets. This work builds upon the foundational research in end-to-end vision based navigation and monocular depth estimation.
