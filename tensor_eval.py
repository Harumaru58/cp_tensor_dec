import numpy as np
import pandas as pd
import os
import tensorly as tl
from tensorly.decomposition import non_negative_parafac

# ===================================================================
# TENSOR DECOMPOSITION EVALUATION FUNCTIONS
# ===================================================================

def evaluate_decomposition_quality(original_tensor, cp_decomposition):
    """
    Evaluate the quality of CP decomposition using reconstruction error metrics.
    
    Args:
        original_tensor (tensorly.tensor): The original 3D tensor
        cp_decomposition (tuple): Tuple of (weights, factors) from CP decomposition
        
    Returns:
        dict: Dictionary containing error metrics and reconstruction tensor
    """
    weights, factors = cp_decomposition
    
    # Reconstruct the tensor from the decomposition
    reconstructed_tensor = tl.cp_to_tensor((weights, factors))
    
    # Calculate reconstruction error
    reconstruction_error = tl.norm(original_tensor - reconstructed_tensor)
    
    # Calculate relative error (normalized by the norm of original tensor)
    original_norm = tl.norm(original_tensor)
    relative_error = reconstruction_error / original_norm
    
    # Calculate explained variance (how much variance is captured)
    total_variance = tl.norm(original_tensor) ** 2
    residual_variance = tl.norm(original_tensor - reconstructed_tensor) ** 2
    explained_variance = 1 - (residual_variance / total_variance)
    
    # Calculate Frobenius norm of the difference
    try:
        frobenius_error = tl.norm(original_tensor - reconstructed_tensor, 'fro')
        relative_frobenius_error = frobenius_error / tl.norm(original_tensor, 'fro')
    except:
        # Fallback calculation if tensorly norm fails
        diff_tensor = original_tensor - reconstructed_tensor
        frobenius_error = np.sqrt(np.sum(diff_tensor ** 2))
        relative_frobenius_error = frobenius_error / np.sqrt(np.sum(original_tensor ** 2))
    
    # Calculate fit score (1 - relative error, similar to R²)
    fit_score = 1 - relative_error
    
    results = {
        'reconstruction_error': float(reconstruction_error),
        'relative_error': float(relative_error),
        'explained_variance': float(explained_variance),
        'frobenius_error': float(frobenius_error),
        'relative_frobenius_error': float(relative_frobenius_error),
        'fit_score': float(fit_score),
        'reconstructed_tensor': reconstructed_tensor
    }
    
    return results


def run_permutation_test(original_tensor, rank, n_permutations=100, random_state=42):
    """
    Run permutation test to assess statistical significance of decomposition.
    
    Args:
        original_tensor (tensorly.tensor): The original 3D tensor
        rank (int): The rank used for decomposition
        n_permutations (int): Number of random permutations to test
        random_state (int): Seed for reproducibility
        
    Returns:
        dict: Dictionary containing permutation test results
    """
    print(f"    Running permutation test with {n_permutations} random shuffles...")
    
    # Get original decomposition quality
    original_weights, original_factors = perform_cp_decomposition_robust(
        original_tensor,
        rank=rank,
        n_iter_max=500,
        random_state=random_state,
        init_method='random',
        tolerance=1e-8,
        normalize_factors=False
    )
    
    original_metrics = evaluate_decomposition_quality(original_tensor, (original_weights, original_factors))
    original_fit_score = original_metrics['fit_score']
    
    # Run permutations
    random_fit_scores = []
    np.random.seed(random_state)
    
    for i in range(n_permutations):
        if i % 20 == 0:  # Progress indicator
            print(f"      Progress: {i}/{n_permutations}")
        
        # Create permuted tensor by shuffling the data
        permuted_tensor = original_tensor.copy()
        permuted_data = permuted_tensor.reshape(-1)  # Flatten
        np.random.shuffle(permuted_data)  # Shuffle
        permuted_tensor = permuted_tensor.reshape(original_tensor.shape)  # Reshape
        
        # Run decomposition on permuted data
        try:
            perm_weights, perm_factors = perform_cp_decomposition_robust(
                permuted_tensor,
                rank=rank,
                n_iter_max=500,
                random_state=None,  # Different seed for each permutation
                init_method='random',
                tolerance=1e-8,
                normalize_factors=False
            )
            
            perm_metrics = evaluate_decomposition_quality(permuted_tensor, (perm_weights, perm_factors))
            random_fit_scores.append(perm_metrics['fit_score'])
            
        except Exception as e:
            print(f"      Warning: Permutation {i} failed: {e}")
            continue
    
    # Calculate p-value
    random_fit_scores = np.array(random_fit_scores)
    p_value = np.mean(random_fit_scores >= original_fit_score)
    
    # Calculate confidence interval
    confidence_interval = np.percentile(random_fit_scores, [2.5, 97.5])
    
    # Calculate effect size (how much better original is than random)
    effect_size = original_fit_score - np.mean(random_fit_scores)
    
    results = {
        'original_fit_score': original_fit_score,
        'random_fit_scores': random_fit_scores,
        'mean_random_fit': np.mean(random_fit_scores),
        'std_random_fit': np.std(random_fit_scores),
        'p_value': p_value,
        'confidence_interval': confidence_interval,
        'effect_size': effect_size,
        'n_permutations': len(random_fit_scores),
        'is_significant': p_value < 0.05
    }
    
    return results


def print_decomposition_metrics(rank, metrics):
    """
    Print formatted decomposition quality metrics.
    
    Args:
        rank (int): The rank used for decomposition
        metrics (dict): Dictionary of error metrics
    """
    print(f"\n--- Decomposition Quality Metrics (Rank {rank}) ---")
    print(f"  Reconstruction Error:        {metrics['reconstruction_error']:.6f}")
    print(f"  Relative Error:              {metrics['relative_error']:.6f}")
    print(f"  Explained Variance:          {metrics['explained_variance']:.6f}")
    print(f"  Frobenius Error:             {metrics['frobenius_error']:.6f}")
    print(f"  Relative Frobenius Error:    {metrics['relative_frobenius_error']:.6f}")
    print(f"  Fit Score:                   {metrics['fit_score']:.6f}")
    print(f"  Fit Percentage:              {metrics['fit_score']*100:.2f}%")


def print_permutation_results(rank, perm_results):
    """
    Print formatted permutation test results.
    
    Args:
        rank (int): The rank used for decomposition
        perm_results (dict): Dictionary of permutation test results
    """
    print(f"\n--- Permutation Test Results (Rank {rank}) ---")
    print(f"  Original Fit Score:         {perm_results['original_fit_score']:.6f}")
    print(f"  Random Fit Score (mean):    {perm_results['mean_random_fit']:.6f}")
    print(f"  Random Fit Score (std):     {perm_results['std_random_fit']:.6f}")
    print(f"  Effect Size:                {perm_results['effect_size']:.6f}")
    print(f"  p-value:                    {perm_results['p_value']:.6f}")
    print(f"  95% CI (Random):            [{perm_results['confidence_interval'][0]:.6f}, {perm_results['confidence_interval'][1]:.6f}]")
    print(f"  Statistical Significance:   {'YES' if perm_results['is_significant'] else 'NO'} (α=0.05)")
    print(f"  Permutations Completed:     {perm_results['n_permutations']}")
    
    # Interpretation
    if perm_results['p_value'] < 0.001:
        significance = "*** (p < 0.001)"
    elif perm_results['p_value'] < 0.01:
        significance = "** (p < 0.01)"
    elif perm_results['p_value'] < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "ns (p ≥ 0.05)"
    
    print(f"  Significance Level:         {significance}")


def perform_cp_decomposition_robust(tensor, rank=3, n_iter_max=1000, random_state=42, 
                                   init_method='random', tolerance=1e-8, normalize_factors=False):
    """
    Perform robust CP decomposition with error handling and multiple initialization attempts.
    
    Args:
        tensor (tensorly.tensor): Input 3D tensor
        rank (int): Rank of the decomposition
        n_iter_max (int): Maximum number of iterations
        random_state (int): Random seed for reproducibility
        init_method (str): Initialization method ('random', 'svd', 'nndsvd')
        tolerance (float): Convergence tolerance
        normalize_factors (bool): Whether to normalize factors after decomposition
        
    Returns:
        tuple: (weights, factors) from CP decomposition
    """
    print(f"    Attempting CP decomposition with rank={rank}, max_iter={n_iter_max}")
    
    # Try different initialization methods if the first one fails
    init_methods = [init_method, 'random', 'nndsvd']
    
    for attempt, init in enumerate(init_methods):
        try:
            print(f"      Attempt {attempt + 1}: Using '{init}' initialization...")
            
            decomp = non_negative_parafac(
                tensor,
                rank=rank,
                init=init,
                n_iter_max=n_iter_max,
                tol=tolerance,
                random_state=random_state,
                normalize_factors=normalize_factors
            )
            
            weights, factors = decomp
            
            # Check if decomposition is valid
            if weights is not None and factors is not None:
                print(f"      Success! Decomposition completed with '{init}' initialization")
                return weights, factors
                
        except Exception as e:
            print(f"      Attempt {attempt + 1} failed: {e}")
            if attempt < len(init_methods) - 1:
                print(f"      Trying next initialization method...")
            continue
    
    # If all attempts failed, raise an error
    raise RuntimeError(f"All CP decomposition attempts failed for rank {rank}")


def save_evaluation_results(rank, original_metrics, normalized_metrics, perm_results=None, output_dir='tensor_evaluation_results'):
    """
    Save all evaluation results to files.
    
    Args:
        rank (int): The rank used for decomposition
        original_metrics (dict): Metrics for original decomposition
        normalized_metrics (dict): Metrics for normalized decomposition
        perm_results (dict): Permutation test results (optional)
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['reconstruction_error', 'relative_error', 'explained_variance', 
                  'frobenius_error', 'relative_frobenius_error', 'fit_score'],
        'Original': [original_metrics[m] for m in ['reconstruction_error', 'relative_error', 
                                                  'explained_variance', 'frobenius_error', 
                                                  'relative_frobenius_error', 'fit_score']],
        'Normalized': [normalized_metrics[m] for m in ['reconstruction_error', 'relative_error', 
                                                      'explained_variance', 'frobenius_error', 
                                                      'relative_frobenius_error', 'fit_score']]
    })
    
    metrics_path = os.path.join(output_dir, f'evaluation_metrics_rank_{rank}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"     Evaluation metrics saved to: {metrics_path}")
    
    # Save permutation results if available
    if perm_results is not None:
        perm_results_path = os.path.join(output_dir, f'permutation_results_rank_{rank}.json')
        import json
        # Convert numpy arrays and other non-serializable types for JSON serialization
        perm_results_json = {}
        for k, v in perm_results.items():
            if isinstance(v, np.ndarray):
                perm_results_json[k] = v.tolist()
            elif isinstance(v, np.bool_):
                perm_results_json[k] = bool(v)
            elif isinstance(v, np.integer):
                perm_results_json[k] = int(v)
            elif isinstance(v, np.floating):
                perm_results_json[k] = float(v)
            else:
                perm_results_json[k] = v
        with open(perm_results_path, 'w') as f:
            json.dump(perm_results_json, f, indent=2)
        print(f"     Permutation results saved to: {perm_results_path}")


# ===================================================================
# MAIN EXECUTION BLOCK (for standalone testing)
# ===================================================================

if __name__ == "__main__":
    print("--- Tensor Evaluation Module ---")
    print("This module contains evaluation functions for tensor decomposition.")
    print("Import these functions into your main tensor decomposition script.")
    print("\nAvailable functions:")
    print("  - evaluate_decomposition_quality()")
    print("  - run_permutation_test()")
    print("  - print_decomposition_metrics()")
    print("  - print_permutation_results()")
    print("  - save_evaluation_results()")
