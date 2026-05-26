# Physics-Informed LSTM for Polymyxin B Pharmacokinetic Modeling in Sepsis Patients

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the complete implementation of the **Physics-Informed Long Short-Term Memory (PI-LSTM)** model described in the manuscript. The model integrates mechanistic pharmacokinetic (PK) principles with a deep learning architecture to enable individualized concentration prediction and PK parameter estimation of polymyxin B in critically ill sepsis patients.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Physics-Informed Design](#physics-informed-design)
- [Loss Function](#loss-function)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Code Structure](#code-structure)
- [Key Hyperparameters](#key-hyperparameters)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Overview

Polymyxin B exhibits complex pharmacokinetics in ICU patients with sepsis, where renal function (eGFR) and body composition fluctuate dynamically. Standard population PK (PopPK) models with fixed covariate structures may fail to capture this variability. The proposed PI-LSTM addresses this by:

1. **Embedding ODE-derived PK knowledge** directly into the neural network as a structural prior, constraining predictions to be physiologically consistent.
2. **Learning individual random effects** (η_CL, η_V) from sparse therapeutic drug monitoring (TDM) observations via an LSTM encoder.
3. **Quantifying predictive uncertainty** through heteroscedastic Gaussian likelihood.
4. **Comparing against a standard two-compartment PopPK model** (NONMEM-style IPRED) as a benchmark.

The entire pipeline — synthetic clinical data generation, model training, evaluation, and figure generation — is self-contained in a single Python script.

---

## Model Architecture

```
Input sequences (TDM observations + covariates)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Input Projection  (Linear → GELU → Dropout)        │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  PK-Aware Positional Encoding                       │
│  • Sinusoidal positional PE                         │
│  • Learnable time-decay encoder  (Δt → exp(-kel·Δt))│
│  • Adaptive gating: gate·PE + (1−gate)·TimeEnc     │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Bidirectional LSTM  (2 layers, hidden=128)         │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  PK-Aware Multi-Head Attention  (4 heads)           │
│  • Dose-weighted attention bias                     │
│  • Pharmacokinetic time-decay on attention scores   │
└────────────────────────┬────────────────────────────┘
                         │
                    Context vector
                         │
          ┌──────────────┴───────────────┐
          ▼                              ▼
┌──────────────────────┐     ┌────────────────────────┐
│  PK Parameter Net    │     │  Residual Predictor     │
│  η_CL, η_V (random  │     │  (neural correction on  │
│  effects via tanh-   │     │   top of physics model) │
│  clamped output)     │     └──────────┬─────────────┘
│                      │               │
│  Fixed covariate     │    ┌──────────▼─────────────┐
│  effects:            │    │  Uncertainty Head       │
│  • eGFR^0.75 → CL   │    │  (log-variance output)  │
│  • Weight^1.0 → V    │    └────────────────────────┘
└──────────┬───────────┘
           │
    CL, V (individual)
           │
           ▼
┌─────────────────────────────────────────────────────┐
│  Physics-Based Concentration Computation            │
│  kel = CL / V                                       │
│  Accumulation factor = 1 / (1 − exp(−kel·τ))       │
│  C_physics = (Dose/V) · Acc · exp(−kel · t_post)   │
└────────────────────────┬────────────────────────────┘
                         │
            C_final = C_physics + ε_residual
```

**Key design choices:**

| Component | Design | Rationale |
|---|---|---|
| Positional encoding | PK-aware (Δt + drug decay) | Irregular sampling intervals in TDM data |
| PK parameter output | Tanh-clamped η × 2ω | Constrains individual effects to ±2 SD of population |
| Covariate effects | Fixed power-law (not learned) | Prevents identifiability collapse with sparse data |
| Concentration prediction | Physics model + small neural residual (×0.3) | Enforces PK structure; residual accounts for model misspecification |
| Uncertainty | Heteroscedastic Gaussian (per-sample log-variance) | Reflects varying observation noise across patients |

---

## Physics-Informed Design

The "physics-informed" components are implemented at **three levels**:

### 1. Structural PK Prior (Hard Embedding)

The forward pass explicitly computes drug concentration using the one-compartment post-infusion equation:

```python
def compute_pk_concentration(self, CL, V, dose, time_since_dose, interval):
    kel = torch.clamp(CL / V, min=0.01, max=2.0)
    acc = 1.0 / (1.0 - torch.exp(-kel * interval) + 1e-8)   # accumulation
    acc = torch.clamp(acc, min=1.0, max=20.0)
    t_post = torch.clamp(time_since_dose - t_inf, min=0.0)  # post-infusion time
    C = (dose / V) * acc * torch.exp(-kel * t_post)
    return torch.clamp(C, min=0.01, max=100.0)
```

The neural network does **not** output concentration directly. It outputs PK parameters (CL, V), and concentration is computed analytically. This is distinct from a pure data-driven approach.

### 2. Population PK Covariate Structure (Semi-parametric)

Individual PK parameters are decomposed following the NONMEM mixed-effects parameterization:

```
log(CL_i) = log(θ_CL) + θ_eGFR · log(eGFR_i / 90) + η_CL_i
log(V_i)  = log(θ_V)  + θ_WT   · log(WT_i / 70)   + η_V_i
```

- `θ_CL`, `θ_V`: population typical values (learnable)
- `θ_eGFR`, `θ_WT`: fixed covariate power coefficients (not learned, set to literature values)
- `η_CL_i`, `η_V_i`: individual random effects predicted by LSTM (tanh-clamped to ±2ω)

### 3. PK-Aware Temporal Encoding

The positional encoder incorporates drug elimination kinetics into sequence encoding:

```python
kel = torch.exp(self.log_kel)                         # learnable kel
pk_decay = torch.exp(-kel * time_deltas).unsqueeze(-1) # mono-exponential decay
time_encoding = time_encoding * pk_decay               # down-weight distant observations
```

This reflects the pharmacokinetic principle that recent TDM samples are more informative than temporally distant ones.

---

## Loss Function

The composite training loss has five terms:

```
L_total = λ_nll · L_NLL
        + λ_physics · L_physics
        + λ_eta_var · L_eta_var
        + λ_kel · L_kel
        + λ_residual · L_residual
```

| Term | Formula | Purpose |
|---|---|---|
| **L_NLL** | Gaussian negative log-likelihood | Primary concentration fitting with uncertainty calibration |
| **L_physics** | Huber(C_physics, C_obs, δ=2.0) | Physics model independently supervised against observations |
| **L_eta_var** | [log(Var(η_CL)) − log(ω²_CL)]² + [...V] | Constrains estimated random-effect variance to match population IIV |
| **L_kel** | ReLU(0.02 − kel)² + ReLU(kel − 0.5)² | Penalizes elimination rate constants outside physiological range |
| **L_residual** | MSE(C_final − C_physics) | Regularizes the neural residual to remain small (physics-first) |

Default weights used in the paper: λ_nll=1.0, λ_physics=1.5, λ_eta_var=0.5, λ_kel=0.5, λ_residual=0.2.

---

## Requirements

```
python >= 3.8
torch >= 2.0
numpy >= 1.24
pandas >= 1.5
scipy >= 1.9
scikit-learn >= 1.2
matplotlib >= 3.6
openpyxl >= 3.0     # for Excel output
```

Install all dependencies:

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib openpyxl
```

GPU acceleration is automatically used if available (CUDA). CPU-only execution is fully supported.

---

## Installation

```bash
git clone https://github.com/<your-username>/PI-LSTM-PolymyxinB.git
cd PI-LSTM-PolymyxinB
pip install -r requirements.txt
```

---

## Quickstart

Run the full pipeline (data simulation → training → evaluation → figures):

```bash
python 6_PI_LSTM.py
```

Expected runtime: ~5–10 minutes on GPU, ~30–60 minutes on CPU (150 epochs, n=1000 simulated patients).

All outputs are saved to the `outputs/` directory.

---

## Code Structure

```
6_PI_LSTM.py
│
├── Configuration
│   └── THETA_CL, THETA_V, OMEGA_CL, OMEGA_V       # Population PK parameters (lines 30-36)
│
├── ClinicalDataSimulator                            # lines 43-261
│   ├── generate_patient_covariates()               # Time-varying eGFR, albumin, CRP, SOFA
│   ├── generate_individual_pk_params()             # Sample η_CL, η_V; compute CL, V
│   ├── pk_ode()                                    # One-compartment ODE (infusion + elimination)
│   ├── simulate_pk_profile()                       # Integrate ODE; add proportional residual error
│   ├── generate_dosing_schedule()                  # Weight- and eGFR-based dosing with adjustments
│   ├── generate_sampling_times()                   # Sparse trough-dominated sampling (TDM-realistic)
│   └── generate_dataset()                          # Assemble full patient dataset as DataFrame
│
├── Data Preprocessing                               # lines 264-358
│   ├── PKDataset                                   # PyTorch Dataset wrapper
│   ├── prepare_sequences()                         # Sliding window sequences (length=3); StandardScaler
│   └── split_by_patient()                          # 70/15/15 train/val/test split (patient-level)
│
├── Model Components                                 # lines 361-705
│   ├── PKAwarePositionalEncoding                   # Sinusoidal PE + learnable PK time-decay
│   ├── PKAwareMultiHeadAttention                   # Standard MHA + dose-weighted + kel-decay bias
│   ├── PKParameterNetworkV4                        # Semi-parametric CL/V estimator with η prediction
│   └── PolymyxinBPKModelV4  (main model)           # Full forward pass: encode → attend → PK params →
│                                                   #   physics concentration → residual → uncertainty
│
├── PKLossV4                                         # lines 708-773
│   └── forward()                                   # NLL + Physics + EtaVar + kel + Residual
│
├── Training & Evaluation                            # lines 776-981
│   ├── EarlyStopping                               # Patience=25, tracks best val loss
│   ├── train_model()                               # AdamW + ReduceLROnPlateau; gradient clipping=1.0
│   └── evaluate_model()                            # MAE, RMSE, R², Pearson r, 95% coverage
│
├── Visualization                                    # lines 983-1626
│   ├── plot_results()                              # Figures 1–6: loss, scatter, BA, PK params, UQ
│   ├── plot_individual_profiles()                  # Figure 7: individual concentration–time profiles
│   └── plot_figure8_comparison()                   # Figure 8: PI-LSTM vs PopPK head-to-head
│
└── main()                                           # lines 1632-1727
```

---

## Key Hyperparameters

| Parameter | Value | Location in code |
|---|---|---|
| n_patients (simulation) | 1000 | `main()`, line 1644 |
| Sequence length | 3 | `prepare_sequences()`, line 1651 |
| Hidden dimension | 128 | `PolymyxinBPKModelV4`, line 1673 |
| LSTM layers | 2 | `PolymyxinBPKModelV4`, line 1673 |
| Attention heads | 4 | `PolymyxinBPKModelV4`, line 1673 |
| Batch size | 64 | `main()`, line 1662 |
| Learning rate | 0.001 | `main()`, line 1685 |
| Weight decay | 1e-4 | `main()`, line 1685 |
| Max epochs | 150 | `main()`, line 1688 |
| Early stopping patience | 25 | `EarlyStopping`, line 810 |
| Gradient clip norm | 1.0 | `train_model()`, line 843 |
| λ_physics | 1.5 | `main()`, line 1679 |
| λ_eta_var | 0.5 | `main()`, line 1680 |
| λ_kel | 0.5 | `main()`, line 1681 |

---

## Outputs

After running `main()`, the `outputs/` directory contains:

| File | Description |
|---|---|
| `polymyxinB_dataset.xlsx` | Simulated patient dataset (all records) |
| `polymyxin_pk_model_v4.pth` | Saved model weights + training history |
| `1_loss.png` | Training and validation loss curves |
| `2_scatter.png` | Observed vs. predicted concentration scatter |
| `3_residuals.png` | Residual distribution plots |
| `4_pk_params.png` | Estimated vs. true CL and V |
| `5_uncertainty.png` | Uncertainty calibration (coverage analysis) |
| `6_attention.png` | Attention weight visualization |
| `Figure7_*.png` | Individual patient concentration–time profiles |
| `Figure8_PILSTM_vs_PopPK.png` | Head-to-head comparison: PI-LSTM vs PopPK (Panel A: observed vs predicted; Panel B: Bland–Altman) |

---

## Reproducibility

All stochastic processes are seeded:

```python
seednum = 42
np.random.seed(seednum)
torch.manual_seed(seednum)
torch.cuda.manual_seed(seednum)  # if GPU available
```

Exact numerical results may vary slightly across hardware/OS due to floating-point non-determinism in cuDNN operations. To enforce fully deterministic GPU execution, add:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Simulation Data Note

The data used in this implementation are **simulated** using the `ClinicalDataSimulator` class, which generates synthetic patient records with:

- Time-varying covariates (eGFR with three trajectory patterns: stable/declining/recovering; albumin; CRP; SOFA score)
- Population PK parameters: θ_CL = 2.5 L/h, θ_V = 35.0 L; IIV: ω_CL = 0.4, ω_V = 0.3
- Sparse trough-dominant sampling design (≈1–2 samples per dosing interval)
- Proportional residual error (CV = 15%) on observed concentrations
- Realistic dosing variability (dose adjustments, interval changes during treatment)

This design was chosen to reflect real-world TDM data characteristics while enabling ground-truth evaluation of PK parameter estimation (true CL and V are known for each simulated patient).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourpaper2025,
  title   = {[Manuscript title]},
  author  = {[Author list]},
  journal = {[Journal name]},
  year    = {2025},
  doi     = {[DOI]}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
