import os
import json
import time
import jax
import jax.numpy as jnp
import optax
from jax import jit, vmap, grad, lax
import numpy as np
from typing import Dict, Tuple, Any, List, Optional, NamedTuple
from tqdm import tqdm

# JAX configuration
jax.config.update("jax_enable_x64", True)

print("🔧 P-TATD-JAX unified module initializing...")

# ============================================================================
# Core Classes and Types
# ============================================================================

class TATDParams(NamedTuple):
    """TATD factor matrix parameters"""
    factors: List[jnp.ndarray]

class PeriodicTATDConfig(NamedTuple):
    """P-TATD configuration"""
    nmode: int
    ndim: Tuple[int, ...]
    tmode: int
    rank: int
    window: int
    periods: Tuple[int, ...]
    periodic_weights: Tuple[float, ...]
    sigma: float = 0.5
    sparse: int = 1

# ============================================================================
# Core Functions
# ============================================================================

@jit
def l2_reg(factor: jnp.ndarray) -> jnp.ndarray:
    """L2 regularization term"""
    return jnp.sum(factor * factor)

def gaussian_smooth(factor: jnp.ndarray, window: int, sigma: float = 0.5) -> jnp.ndarray:
    """Apply Gaussian smoothing"""
    def apply_smoothing():
        time_steps, rank = factor.shape
        
        # Generate Gaussian kernel
        half_window = (window - 1) // 2
        coords = jnp.arange(-half_window, half_window + 1)
        kernel = jnp.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / jnp.sum(kernel)
        
        def smooth_column(col):
            # Convolution with padding
            padded = jnp.pad(col, half_window, mode='edge')
            return jnp.convolve(padded, kernel, mode='valid')
        
        # Apply vmap to each column
        smoothed = vmap(smooth_column, in_axes=1, out_axes=1)(factor)
        return smoothed
    
    def no_smoothing():
        return factor
    
    return jax.lax.cond(window > 1, apply_smoothing, no_smoothing)

@jit
def krprod(factors: List[jnp.ndarray], indices: jnp.ndarray) -> jnp.ndarray:
    """Khatri-Rao product computation for sparse tensors"""
    rank = factors[0].shape[1]
    nnz = indices.shape[1]
    
    # Initialize
    result = jnp.ones((nnz, rank))
    
    # Compute element-wise product for each mode
    for mode, factor in enumerate(factors):
        result = result * factor[indices[mode], :]
    
    # Sum along rows
    return jnp.sum(result, axis=1)

@jit
def tatd_forward(params: TATDParams, indices: jnp.ndarray) -> jnp.ndarray:
    """TATD forward pass"""
    return krprod(params.factors, indices)

def periodic_regularization(factor: jnp.ndarray,
                           periods: Tuple[int, ...],
                           weights: Tuple[float, ...]) -> jnp.ndarray:
    """Periodic regularization term computation"""
    time_steps, rank = factor.shape
    periodic_loss = 0.0

    # Compute for each period
    for period, weight in zip(periods, weights):
        def compute_period_loss():
            valid_indices = time_steps - period
            current = factor[:valid_indices, :]
            shifted = factor[period:period+valid_indices, :]
            periodic_diff = jnp.sum((current - shifted) ** 2)
            return weight * periodic_diff

        def skip_period():
            return 0.0

        period_loss = jax.lax.cond(
            period < time_steps,
            compute_period_loss,
            skip_period
        )
        periodic_loss += period_loss

    return periodic_loss

def smooth_reg_periodic(factor: jnp.ndarray, 
                       smoothed: jnp.ndarray,
                       density: jnp.ndarray, 
                       sparse: int) -> jnp.ndarray:
    """P-TATD smoothing regularization term"""
    diff = smoothed - factor
    sloss = diff * diff
    
    def apply_sparse():
        return sloss * density.reshape(-1, 1)
    
    def no_sparse():
        return sloss
    
    result_sloss = jax.lax.cond(
        sparse == 1,
        apply_sparse,
        no_sparse
    )
    
    return jnp.sum(result_sloss)

@jit
def evaluate(params: TATDParams, data: Dict[str, jnp.ndarray]) -> Tuple[float, float]:
    """Model evaluation"""
    rec = tatd_forward(params, data['indices'])
    diff = rec - data['values']
    
    rmse = jnp.sqrt(jnp.mean(diff ** 2))
    mae = jnp.mean(jnp.abs(diff))
    
    return rmse, mae

def initialize_periodic_factors(config: PeriodicTATDConfig, 
                               key: jax.random.PRNGKey) -> TATDParams:
    """Initialize P-TATD factor matrices"""
    keys = jax.random.split(key, config.nmode)
    factors = []
    
    for mode in range(config.nmode):
        shape = (config.ndim[mode], config.rank)
        # Xavier initialization
        scale = jnp.sqrt(2.0 / sum(shape))
        factor = jax.random.normal(keys[mode], shape) * scale
        factors.append(factor)
    
    return TATDParams(factors=factors)

def create_periodic_config(nmode: int, ndim: Tuple[int, ...], tmode: int, 
                          rank: int, window: int, periods: List[int], 
                          periodic_weights: List[float], sigma: float = 0.5, 
                          sparse: int = 1) -> PeriodicTATDConfig:
    """Create P-TATD configuration helper function"""
    return PeriodicTATDConfig(
        nmode=nmode,
        ndim=ndim,
        tmode=tmode,
        rank=rank,
        window=window,
        periods=tuple(periods),
        periodic_weights=tuple(periodic_weights),
        sigma=sigma,
        sparse=sparse
    )

# ============================================================================
# Loss and Gradient Functions
# ============================================================================

def compute_regularization_components(params: TATDParams,
                                    config: PeriodicTATDConfig,
                                    density: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute regularization term components"""
    time_factor = params.factors[config.tmode]
    
    # Smoothing regularization
    smoothed = gaussian_smooth(time_factor, config.window, config.sigma)
    smooth_loss = smooth_reg_periodic(time_factor, smoothed, density, config.sparse)
    
    # Periodic regularization
    periodic_loss = periodic_regularization(time_factor, config.periods, config.periodic_weights)
    
    # L2 regularization for non-temporal modes
    l2_loss = sum(l2_reg(params.factors[n]) 
                  for n in range(config.nmode) if n != config.tmode)
    
    return smooth_loss, periodic_loss, l2_loss

def compute_periodic_tatd_loss(params: TATDParams,
                              train_data: Dict[str, jnp.ndarray],
                              config: PeriodicTATDConfig,
                              penalty_smooth: float,
                              penalty_periodic: float,
                              penalty_l2: float,
                              density: jnp.ndarray) -> jnp.ndarray:
    """P-TATD loss function computation"""
    indices = train_data['indices']
    values = train_data['values']
    
    # Reconstruction error
    rec = tatd_forward(params, indices)
    rec_loss = jnp.sum((rec - values) ** 2)
    
    # Regularization terms computation
    smooth_loss, periodic_loss, l2_loss = compute_regularization_components(
        params, config, density
    )
    
    # Total loss
    total_loss = (rec_loss + 
                  penalty_smooth * smooth_loss + 
                  penalty_periodic * periodic_loss + 
                  penalty_l2 * l2_loss)
    
    return total_loss

def compute_loss_components(params: TATDParams,
                           train_data: Dict[str, jnp.ndarray],
                           config: PeriodicTATDConfig,
                           density: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute individual loss components (for debugging)"""
    indices = train_data['indices']
    values = train_data['values']
    
    # Reconstruction error
    rec = tatd_forward(params, indices)
    rec_loss = jnp.sum((rec - values) ** 2)
    
    # Regularization terms
    smooth_loss, periodic_loss, l2_loss = compute_regularization_components(
        params, config, density
    )
    
    return rec_loss, smooth_loss, periodic_loss, l2_loss

def compute_periodic_gradients(params: TATDParams,
                              train_data: Dict[str, jnp.ndarray],
                              config: PeriodicTATDConfig,
                              penalty_smooth: float,
                              penalty_periodic: float,
                              penalty_l2: float,
                              density: jnp.ndarray) -> TATDParams:
    """Gradient computation function"""
    def loss_fn(p):
        return compute_periodic_tatd_loss(
            p, train_data, config, penalty_smooth, penalty_periodic, penalty_l2, density
        )
    
    grad_fn = jax.grad(loss_fn)
    return grad_fn(params)

# ============================================================================
# Data Loading Functions
# ============================================================================

def calculate_time_sparsity(data: Dict[str, jnp.ndarray], tmode: int) -> jnp.ndarray:
    """Calculate time sparsity"""
    time_indices = data['indices'][tmode]
    max_time = int(jnp.max(time_indices)) + 1
    
    counts = jnp.zeros(max_time)
    for t in range(max_time):
        count_t = jnp.sum(time_indices == t)
        counts = counts.at[t].set(count_t)
    
    min_count = jnp.min(counts)
    max_count = jnp.max(counts)
    
    def apply_normalization():
        density = (0.999 - 0.001) * (counts - min_count) / (max_count - min_count) + 0.001
        return density
    
    def uniform_density():
        return jnp.ones_like(counts) * 0.5
    
    density = jax.lax.cond(
        max_count > min_count,
        apply_normalization,
        uniform_density
    )
    
    return 1.0 - density

def load_direct_npy_dataset(data_path):
    """Load directly placed npy format data"""
    print("📥 Loading direct npy data:")
    print(f"Data path: {data_path}")

    # Check required files
    required_files = {
        'train': ['train_idxs.npy', 'train_vals.npy'],
        'valid': ['valid_idxs.npy', 'valid_vals.npy'],
        'test': ['test_idxs.npy', 'test_vals.npy']
    }

    dataset = {}

    try:
        # Load each split data
        for split, files in required_files.items():
            idx_file, val_file = files
            idx_path = os.path.join(data_path, idx_file)
            val_path = os.path.join(data_path, val_file)

            if os.path.exists(idx_path) and os.path.exists(val_path):
                print(f"  ✅ Loading {split} data...")

                # Load as numpy arrays
                indices_np = np.load(idx_path)  # (nnz, nmode)
                values_np = np.load(val_path)   # (nnz,)

                print(f"    Index shape: {indices_np.shape}")
                print(f"    Values shape: {values_np.shape}")
                print(f"    Value range: [{np.min(values_np):.3f}, {np.max(values_np):.3f}]")

                # Convert to JAX format (transpose to COO format)
                indices_jax = jnp.array(indices_np.T, dtype=jnp.int32)  # (nmode, nnz)
                values_jax = jnp.array(values_np, dtype=jnp.float64)     # (nnz,)

                dataset[split] = {
                    'indices': indices_jax,
                    'values': values_jax
                }

                print(f"    JAX conversion complete: indices {indices_jax.shape}, values {values_jax.shape}")
            else:
                print(f"  ❌ {split} data files not found")

        # Calculate metadata
        if 'train' in dataset:
            train_indices = dataset['train']['indices']

            # Calculate tensor dimensions
            max_indices = jnp.max(train_indices, axis=1)
            ndim = tuple(int(m) + 1 for m in max_indices)
            nmode = len(ndim)

            print(f"\n📊 Dataset information:")
            print(f"  Tensor shape: {ndim}")
            print(f"  Number of modes: {nmode}")
            print(f"  Total elements: {np.prod(ndim)}")

            # Calculate density for each split
            total_elements = np.prod(ndim)
            for split in dataset:
                n_entries = len(dataset[split]['values'])
                density = n_entries / total_elements
                print(f"  {split}: {n_entries} entries (density: {density:.4f})")

            # Set metadata
            dataset.update({
                'name': 'direct_npy_dataset',
                'ndim': ndim,
                'nmode': nmode,
                'tmode': 0,  # Default time mode (first mode)
            })

            # Calculate time sparsity
            time_indices = train_indices[0]  # Time mode
            max_time = int(jnp.max(time_indices)) + 1

            # Count elements for each time slice
            counts = jnp.zeros(max_time)
            for t in range(max_time):
                count_t = jnp.sum(time_indices == t)
                counts = counts.at[t].set(count_t)

            # Calculate sparsity
            min_count = jnp.min(counts)
            max_count = jnp.max(counts)

            if max_count > min_count:
                density = (0.999 - 0.001) * (counts - min_count) / (max_count - min_count) + 0.001
            else:
                density = jnp.ones_like(counts) * 0.5

            dataset['ts'] = 1.0 - density

            print(f"  Time mode: {dataset['tmode']}")
            print(f"  Time steps: {max_time}")

            print(f"\n✅ Dataset loading complete!")
            return dataset

        else:
            print("❌ Training data not found")
            return None

    except Exception as e:
        print(f"❌ Data loading error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_custom_config(dataset):
    """Create dataset-specific P-TATD configuration"""
    if dataset is None:
        return None

    print("\n⚙️ Custom P-TATD configuration:")

    ndim = dataset['ndim']
    nmode = dataset['nmode']
    tmode = dataset['tmode']

    # Determine settings based on data characteristics
    time_steps = ndim[tmode]
    total_elements = np.prod(ndim)

    print(f"  Tensor shape: {ndim}")
    print(f"  Time steps: {time_steps}")

    # Rank optimized for small data
    if total_elements < 10000:
        rank = 3
        print(f"  Small data → Rank: {rank}")
    elif total_elements < 50000:
        rank = 5
        print(f"  Medium-small data → Rank: {rank}")
    else:
        rank = 8
        print(f"  Medium data → Rank: {rank}")

    # Special settings for 24 time steps
    periods = []
    periodic_weights = []

    if time_steps == 24:
        # 24-hour data case
        periods = [8, 12]  # 8-hour and 12-hour cycles
        periodic_weights = [1.0, 1.5]
        print(f"  24-hour data → 8-hour & 12-hour cycles")
    elif time_steps >= 24:
        periods = [24]
        periodic_weights = [2.0]
        if time_steps >= 168:
            periods.append(168)
            periodic_weights.append(1.0)
        print(f"  Long-term data → {periods}-hour cycles")
    else:
        # Short-term data case
        periods = [max(2, time_steps // 3)]
        periodic_weights = [1.0]
        print(f"  Short-term data → {periods[0]}-hour cycle")

    # Adjust window size
    window = min(5, max(3, time_steps // 6))
    print(f"  Window size: {window}")

    # Create configuration
    config = create_periodic_config(
        nmode=nmode,
        ndim=ndim,
        tmode=tmode,
        rank=rank,
        window=window,
        periods=periods,
        periodic_weights=periodic_weights,
        sigma=0.8,  # Slightly smaller sigma
        sparse=1
    )

    print(f"✅ Custom configuration creation complete")
    return config

# ============================================================================
# Training Functions
# ============================================================================

def train_periodic_tatd_pure_gradient(
    dataset: Dict[str, Any],
    config: PeriodicTATDConfig,
    penalty_smooth: float = 100.0,
    penalty_periodic: float = 10.0,
    penalty_l2: float = 1.0,
    lr: float = 0.01,
    max_iters: int = 500,
    verbose: bool = True
) -> Tuple[TATDParams, float]:
    """Pure gradient-based P-TATD training"""
    if verbose:
        print("=== Pure Gradient P-TATD Training ===")

    start_time = time.time()

    # Initialization
    key = jax.random.PRNGKey(1234)
    params = initialize_periodic_factors(config, key)
    density = calculate_time_sparsity(dataset['train'], config.tmode)

    # Adam optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(params, opt_state):
        grads = compute_periodic_gradients(
            params, dataset['train'], config,
            penalty_smooth, penalty_periodic, penalty_l2, density
        )

        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    # Training loop
    for iteration in range(max_iters):
        params, opt_state = update_step(params, opt_state)

        if iteration % 10 == 0 and verbose:
            loss = compute_periodic_tatd_loss(
                params, dataset['train'], config,
                penalty_smooth, penalty_periodic, penalty_l2, density
            )
            print(f"Iter {iteration}: Loss = {loss:.6f}")

    training_time = time.time() - start_time

    if verbose:
        print(f"Pure gradient training completed in {training_time:.2f}s")

    return params, training_time

def train_periodic_tatd_gradient_optimized(
    dataset: Dict[str, Any],
    config: PeriodicTATDConfig,
    penalty_smooth: float = 100.0,
    penalty_periodic: float = 10.0,
    penalty_l2: float = 1.0,
    lr: float = 0.01,
    max_iters: int = 1000,
    patience: int = 10,
    eval_interval: int = 5,
    optimization_strategy: str = 'simultaneous',
    verbose: bool = True
) -> Tuple[TATDParams, Dict[str, List[float]]]:
    """Gradient-optimized P-TATD training"""

    if verbose:
        print("=== P-TATD Gradient-Optimized Training ===")
        print(f"Strategy: {optimization_strategy}")
        print(f"Max iterations: {max_iters}")
        print(f"Learning rate: {lr}")

    # Initialization
    key = jax.random.PRNGKey(1234)
    params = initialize_periodic_factors(config, key)
    density = calculate_time_sparsity(dataset['train'], config.tmode)

    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def simultaneous_update_step(params, opt_state):
        grads = compute_periodic_gradients(
            params, dataset['train'], config,
            penalty_smooth, penalty_periodic, penalty_l2, density
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # Training history
    history = {
        'train_rmse': [], 'val_rmse': [], 'train_mae': [], 'val_mae': [],
        'total_loss': [], 'rec_loss': [], 'smooth_loss': [],
        'periodic_loss': [], 'l2_loss': []
    }

    # Early stopping
    best_val_rmse = float('inf')
    patience_counter = 0
    best_params = params

    start_time = time.time()

    try:
        iterator = tqdm(range(max_iters), desc="P-TATD Training") if verbose else range(max_iters)

        for iteration in iterator:
            # Parameter update
            params, opt_state = simultaneous_update_step(params, opt_state)

            # Periodic evaluation
            if iteration % eval_interval == 0:
                # RMSE/MAE evaluation
                train_rmse, train_mae = evaluate(params, dataset['train'])
                val_rmse, val_mae = evaluate(params, dataset['valid'])

                # Loss evaluation
                rec_loss, smooth_loss, periodic_loss, l2_loss = compute_loss_components(
                    params, dataset['train'], config, density
                )

                total_loss = (rec_loss +
                            penalty_smooth * smooth_loss +
                            penalty_periodic * periodic_loss +
                            penalty_l2 * l2_loss)

                # Add to history
                history['train_rmse'].append(float(train_rmse))
                history['val_rmse'].append(float(val_rmse))
                history['train_mae'].append(float(train_mae))
                history['val_mae'].append(float(val_mae))
                history['total_loss'].append(float(total_loss))
                history['rec_loss'].append(float(rec_loss))
                history['smooth_loss'].append(float(smooth_loss))
                history['periodic_loss'].append(float(periodic_loss))
                history['l2_loss'].append(float(l2_loss))

                # Early stopping check
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_params = params
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Progress display
                if verbose and hasattr(iterator, 'set_description'):
                    iterator.set_description(
                        f'RMSE: {train_rmse:.4f}/{val_rmse:.4f} '
                        f'Loss: {total_loss:.2e}'
                    )

                # Early stopping decision
                if patience_counter >= patience and iteration > 50:
                    if verbose:
                        print(f"Early stopping at iteration {iteration}")
                    break

                # NaN check
                if jnp.isnan(train_rmse) or jnp.isnan(val_rmse):
                    if verbose:
                        print(f"NaN detected at iteration {iteration}. Stopping training.")
                    break

    except KeyboardInterrupt:
        if verbose:
            print("\nTraining interrupted by user")

    training_time = time.time() - start_time

    if verbose:
        final_train_rmse, final_train_mae = evaluate(best_params, dataset['train'])
        final_val_rmse, final_val_mae = evaluate(best_params, dataset['valid'])

        print("\n=== P-TATD Training Complete ===")
        print(f"Total training time: {training_time:.2f}s")
        print(f"Final Train RMSE: {final_train_rmse:.5f}")
        print(f"Final Val RMSE: {final_val_rmse:.5f}")
        print(f"Iterations completed: {len(history['train_rmse']) * eval_interval}")

    return best_params, history

# ============================================================================
# Easy Run Functions
# ============================================================================

def setup_and_run_tatd(
    data_path: str,
    test_type: str = 'fast',
    max_iters: Optional[int] = None,
    lr: float = 0.01,
    verbose: bool = True
) -> Tuple[Any, Any, Any]:
    """
    One-stop execution from data loading to training and evaluation
    """
    
    if verbose:
        print("🚀 P-TATD-JAX one-stop execution start!")
        print("=" * 50)
    
    # Step 1: Data loading
    if verbose:
        print("📥 Step 1: Data loading")
    
    dataset = load_direct_npy_dataset(data_path)
    if dataset is None:
        raise RuntimeError("Data loading failed")
    
    # Step 2: Configuration creation
    if verbose:
        print("\n⚙️ Step 2: Configuration creation")
    
    config = create_custom_config(dataset)
    if config is None:
        raise RuntimeError("Configuration creation failed")
    
    # Step 3: Training execution
    if verbose:
        print("\n🎯 Step 3: Training execution")
    
    # Determine iteration count
    if max_iters is None:
        if test_type == 'fast':
            max_iters = 15
            lr = 0.05
        elif test_type == 'medium':
            max_iters = 40
            lr = 0.03
        else:  # full
            max_iters = 80
            lr = 0.01
    
    if verbose:
        print(f"  Test type: {test_type}")
        print(f"  Max iterations: {max_iters}")
        print(f"  Learning rate: {lr}")
    
    # Execute training
    params, training_time = train_periodic_tatd_pure_gradient(
        dataset=dataset,
        config=config,
        penalty_smooth=50.0,
        penalty_periodic=5.0,
        penalty_l2=1.0,
        lr=lr,
        max_iters=max_iters,
        verbose=verbose
    )
    
    # Step 4: Evaluation
    if verbose:
        print("\n📊 Step 4: Evaluation results")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Per iteration: {training_time/max_iters*1000:.1f}ms")
        
        for split in ['train', 'valid', 'test']:
            if split in dataset:
                rmse, mae = evaluate(params, dataset[split])
                print(f"  {split:5s}: RMSE={rmse:.6f}, MAE={mae:.6f}")
        
        # GPU usage check
        print(f"  Computing device: {jax.default_backend()}")
        
        # Parameter information
        total_params = sum(f.size for f in params.factors)
        print(f"  Total parameters: {total_params:,}")
        
        print("\n✅ P-TATD-JAX execution complete!")
    
    return dataset, config, params

def run_with_history(
    data_path: str,
    max_iters: int = 30,
    lr: float = 0.01,
    eval_interval: int = 3,
    verbose: bool = True
) -> Tuple[Any, Any, Any, Dict]:
    """Execute P-TATD with history tracking"""
    
    if verbose:
        print("🚀 P-TATD-JAX execution with history start!")
        print("=" * 50)
    
    # Data loading and configuration creation
    dataset = load_direct_npy_dataset(data_path)
    if dataset is None:
        raise RuntimeError("Data loading failed")
    
    config = create_custom_config(dataset)
    if config is None:
        raise RuntimeError("Configuration creation failed")
    
    # Training with history
    params, history = train_periodic_tatd_gradient_optimized(
        dataset=dataset,
        config=config,
        penalty_smooth=100.0,
        penalty_periodic=10.0,
        penalty_l2=1.0,
        lr=lr,
        max_iters=max_iters,
        eval_interval=eval_interval,
        optimization_strategy='simultaneous',
        verbose=verbose
    )
    
    if verbose:
        print("\n📊 Final evaluation:")
        for split in ['train', 'valid', 'test']:
            if split in dataset:
                rmse, mae = evaluate(params, dataset[split])
                print(f"  {split}: RMSE={rmse:.6f}, MAE={mae:.6f}")
        
        print("\n✅ Execution with history complete!")
    
    return dataset, config, params, history

def quick_demo(data_path: str) -> bool:
    """Quick demo execution"""
    try:
        print("🎮 P-TATD-JAX Quick Demo!")
        print("=" * 40)
        
        dataset, config, params = setup_and_run_tatd(
            data_path=data_path,
            test_type='fast',
            verbose=True
        )
        
        print("\n🎉 Demo successful! P-TATD-JAX is working properly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Easy-to-use aliases
def run_tatd(data_path: str, **kwargs):
    """Alias: setup_and_run_tatd"""
    return setup_and_run_tatd(data_path, **kwargs)

def run_tatd_full(data_path: str, **kwargs):
    """Alias: run_with_history"""
    return run_with_history(data_path, **kwargs)

# JAX initialization check
def check_setup():
    """Setup verification"""
    print("🔧 P-TATD-JAX setup verification:")
    print(f"  JAX version: {jax.__version__}")
    print(f"  Available devices: {jax.devices()}")
    print(f"  Default backend: {jax.default_backend()}")
    
    # Simple test
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    print(f"  JAX operation test: sum([1,2,3]) = {y}")
    print("✅ Setup verification complete!")

if __name__ == "__main__":
    # When this file is executed directly
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        quick_demo(data_path)
    else:
        print("Usage: python unified_tatd.py <data_path>")
        print("Or:")
        print("from unified_tatd import run_tatd")
        print("dataset, config, params = run_tatd('/path/to/data')")
