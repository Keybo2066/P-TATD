# P-TATD

# P-TATD-JAX: Periodic Time-Aware Tensor Decomposition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-orange.svg)](https://github.com/google/jax)

A high-performance JAX implementation of Periodic Time-Aware Tensor Decomposition (P-TATD) that explicitly captures periodic patterns in temporal data. This library provides efficient tensor decomposition with JIT compilation, automatic differentiation, and GPU acceleration.



## 📚 Background

In many scientific fields, including natural and social sciences, understanding features and relationships among variables from collected data is crucial. For example, in urban air pollution analysis, observation locations (where), observation times (when), and types of observed pollutants (what) interact to form complex spatiotemporal patterns.

Traditional tensor decomposition methods often fail to adequately capture temporal characteristics, particularly **periodic variations**. Real-world data frequently exhibit clear periodicities such as:
- **Diurnal variations** (daily cycles)
- **Weekly patterns** (day-of-week effects) 
- **Seasonal fluctuations** (monthly/yearly cycles)

P-TATD extends the Time-Aware Tensor Decomposition (TATD) framework to explicitly model these periodic features, enabling more precise analysis and prediction.

## 🔬 Mathematical Formulation

### Objective Function

P-TATD solves the following optimization problem:

```
min     Σ(x_ijk - Σ t_ir u_jr v_kr)² 
T,U,V   (i,j,k)∈Ω    r=1

        + λ_s Σ β_i ||t_i - t̃_i||²     [Smoothing Regularization]
              i=1

        + λ_p Σ α_m Σ ||t_i - t_{i+P_m}||²  [Periodic Regularization]
              m=1   i=1

        + λ_r (||U||²_F + ||V||²_F)     [L2 Regularization]
```

### Notation

| Symbol | Description |
|--------|-------------|
| **X** ∈ ℝ^(I×J×K) | Third-order tensor |
| Ω | Set of observed entry indices |
| I | Temporal mode dimension |
| J, K | Non-temporal mode dimensions |
| R | Tensor rank |
| **T** ∈ ℝ^(I×R) | Temporal factor matrix |
| **U** ∈ ℝ^(J×R) | Second mode factor matrix |
| **V** ∈ ℝ^(K×R) | Third mode factor matrix |
| **t**_i ∈ ℝ^R | i-th row vector of **T** |

### Regularization Terms

#### 1. Smoothing Regularization
Encourages temporal smoothness with sparsity-adaptive weighting:

```
β_i = 1 - d_i
d_i = (0.999 - 0.001) × (ω_i - ω_min)/(ω_max - ω_min) + 0.001
```

where ω_i is the number of non-zero entries at time slice i.

The smoothed factor t̃_i is computed as:
```
t̃_i = Σ w(i,i_s) t_{i_s}
     i_s∈μ(i,S)

w(i,i_s) = K(i,i_s) / Σ K(i,i_s')
                      i_s'∈μ(i,S)

K(i,i_s) = exp(-(i-i_s)²/(2σ²))
```

#### 2. Periodic Regularization
Enforces periodic patterns by minimizing differences between factor vectors separated by period lengths:

```
Σ ||t_i - t_{i+P_m}||²
i=1
```

This term encourages the model to learn recurring cyclical behaviors (e.g., daily, weekly patterns).

#### 3. Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| λ_s | Smoothing penalty weight | 10-1000 |
| λ_p | Periodic penalty weight | 1-100 |
| λ_r | L2 penalty weight | 0.1-10 |
| P_m | Period lengths | [24, 168, ...] |
| α_m | Period-specific weights | [0.5, 2.0] |
| S | Smoothing window size | 3-7 |
| σ | Gaussian kernel width | 0.5-1.0 |



## 📋 Data Format

P-TATD-JAX expects sparse tensors in COO (Coordinate) format:

```
data/
├── train_idxs.npy    # Training indices (nnz, nmode)
├── train_vals.npy    # Training values (nnz,)
├── valid_idxs.npy    # Validation indices
├── valid_vals.npy    # Validation values
├── test_idxs.npy     # Test indices
└── test_vals.npy     # Test values
```

### Data Format Details

- **Indices files** (`*_idxs.npy`): Shape `(nnz, nmode)` where `nnz` is the number of non-zero entries and `nmode` is the number of tensor modes
- **Values files** (`*_vals.npy`): Shape `(nnz,)` containing the actual tensor values
- **Index convention**: Zero-based indexing for all modes
- **Supported data types**: `numpy.int32` for indices, `numpy.float64` for values




### Configuration

```python
from p_tatd import create_periodic_config

# configuration for specific use cases
config = create_periodic_config(
    nmode=3,                          # Number of modes
    ndim=(168, 20, 4),               # Tensor dimensions
    tmode=0,                         # Time mode index
    rank=8,                          # Tensor rank
    window=5,                        # Smoothing window
    periods=[24, 168],               # Daily and weekly cycles
    periodic_weights=[2.0, 1.0],     # Period-specific weights
    sigma=0.7,                       # Gaussian kernel width
    sparse=1                         # Enable sparsity adaptation
)
```

## ⚙️ Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `rank` | Tensor rank | Auto-detected | 3-20 |
| `periods` | Period lengths | Auto-detected | [2, max_time] |
| `window` | Smoothing window | Auto-sized | 3-7 |
| `penalty_smooth` | Smoothing weight | 100.0 | 10-1000 |
| `penalty_periodic` | Periodic weight | 10.0 | 1-100 |
| `penalty_l2` | L2 weight | 1.0 | 0.1-10 |
| `sigma` | Gaussian width | 0.5 | 0.3-1.0 |






```



## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



---
