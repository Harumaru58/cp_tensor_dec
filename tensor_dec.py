import os
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from sklearn.preprocessing import MinMaxScaler

# Import evaluation functions
from tensor_eval import (
    evaluate_decomposition_quality,
    run_permutation_test,
    print_decomposition_metrics,
    print_permutation_results,
    save_evaluation_results
)

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def load_and_create_tensor(data_path, feature_list):
    """
    Loads raw data, creates a 3D tensor, and scales it for non-negative decomposition.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at '{data_path}'. Please run the data generation script first.")
    print(f"Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)

    # Determine tensor dimensions
    n_agents = df['agent_id'].nunique()
    n_trials = df['trial'].nunique()
    n_features = len(feature_list)

    # Sort values to ensure correct reshaping order
    df_sorted = df.sort_values(['agent_id', 'trial'])
    
    # Reshape the data into a 3D numpy array
    raw_tensor = df_sorted[feature_list].values.reshape(n_agents, n_trials, n_features)

    # --- Preprocessing for Non-Negative Decomposition ---
    # Reshape for scaling: (samples, features) where a "sample" is one trial from one agent
    tensor_2d = raw_tensor.reshape(-1, n_features)
    
    # Use MinMaxScaler to ensure all values are in the [0, 1] range, preserving non-negativity
    scaler = MinMaxScaler()
    scaled_tensor_2d = scaler.fit_transform(tensor_2d)
    
    # Reshape back to 3D and convert to a tensorly tensor
    scaled_tensor = tl.tensor(scaled_tensor_2d.reshape(n_agents, n_trials, n_features))
    
    return scaled_tensor, feature_list



def perform_cp_decomposition(tensor, rank=3, n_iter_max=5000, random_state=42):
    """Perform non-negative CP decomposition with error handling"""
    try:
        decomp = non_negative_parafac(
            tensor,
            init="random",
            rank=rank,
            n_iter_max=n_iter_max,
            tol=1e-8, 
            random_state=random_state,
            normalize_factors=False
        )
        
        print(f"CP decomposition completed for rank {rank}!")
        return decomp
    except Exception as e:
        print(f"CP decomposition failed for rank {rank}: {e}")
        raise


def manual_l1_normalize(cp_decomposition, min_norm=1e-12):
    """
    Normalize CP factors using L1 norm and adjust lambda weights accordingly.
    
    This function:
    1. Normalizes each factor matrix so that each column sums to 1 (L1 norm)
    2. Adjusts the lambda weights to preserve the mathematical relationship
    3. Makes factors interpretable as probability distributions
    
    Mathematical relationship:
    λ_new = λ_original × ||U||₁ × ||V||₁ × ||W||₁
    where ||·||₁ is the L1 norm of each factor matrix
    """
    weights = cp_decomposition[0].copy()
    factors = [f.copy() for f in cp_decomposition[1]]
    
    for idx, factor in enumerate(factors):
        l1_norms = np.sum(np.abs(factor), axis=0)
        l1_norms[l1_norms < min_norm] = 1.0 # Avoid division by zero
        factors[idx] = factor / l1_norms[np.newaxis, :]
        weights *= l1_norms
    
    return (weights, factors)











if __name__ == "__main__":
    # --- 1. CONFIGURATION ---
    DATA_PATH = 'datasets/observable_bandit_data_1000agents.csv'
    OUTPUT_DIR = 'tensor_decomposition_results'
    
    # Define the raw, observable features to include in the tensor
    FEATURES_TO_USE = ['action', 'reward']
    
    # Define the list of ranks to run the decomposition for
    RANKS_TO_RUN = [3, 5, 6]
    
    # Permutation test configuration
    PERMUTATION_TESTS = False          # Disabled to see reconstruction errors for both ranks
    N_PERMUTATIONS = 100             # Number of random shuffles (higher = more accurate but slower)
    ALPHA = 0.05                     # Significance level
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. LOAD AND PREPARE THE TENSOR ---
    print("--- [Step 1] Preparing Data Tensor ---")
    data_tensor, feature_names = load_and_create_tensor(DATA_PATH, FEATURES_TO_USE)

    # --- 3. RUN DECOMPOSITION FOR EACH RANK ---
    print("\n--- [Step 2] Running Non-Negative CP Decomposition ---")
    for rank in RANKS_TO_RUN:
        print(f"\nProcessing for RANK = {rank}...")
        
        # Perform the decomposition using our function
        weights, factors = perform_cp_decomposition(
            data_tensor,
            rank=rank,
            n_iter_max=5000,
            random_state=42
        )
        
        #Normalize factors for better interpretation
        #_, normalized_factors = manual_l1_normalize((weights, factors))

        # Extract the individual factor matrices
        agent_factors, trial_factors, feature_factors = factors

        print(f"  -> Decomposition complete. Extracted factors with shapes:")
        print(f"     Agents:   {agent_factors.shape}")
        print(f"     Trials:   {trial_factors.shape}")
        print(f"     Features: {feature_factors.shape}")

        # --- 4. EVALUATE DECOMPOSITION QUALITY ---
        print("  -> Evaluating decomposition quality...")
        
        # Evaluate the original decomposition (before normalization)
        original_metrics = evaluate_decomposition_quality(data_tensor, (weights, factors))
        
        # Evaluate the normalized decomposition
        #normalized_metrics = evaluate_decomposition_quality(data_tensor, (weights, normalized_factors))
        
        # Print metrics
        print_decomposition_metrics(rank, original_metrics)

        # --- 5. RUN PERMUTATION TESTS (if enabled) ---
        if PERMUTATION_TESTS:
            print("  -> Running permutation tests for statistical significance...")
            perm_results = run_permutation_test(
                data_tensor, 
                rank, 
                n_permutations=N_PERMUTATIONS,
                random_state=42
            )
            print_permutation_results(rank, perm_results)
            
            # Save permutation results
            perm_results_path = os.path.join(OUTPUT_DIR, f'permutation_results_rank_{rank}.json')
            import json
            # Convert numpy arrays to lists for JSON serialization
            perm_results_json = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in perm_results.items()}
            with open(perm_results_path, 'w') as f:
                json.dump(perm_results_json, f, indent=2)
            print(f"     Permutation results saved to: {perm_results_path}")
            
            # Save all evaluation results
            save_evaluation_results(rank, original_metrics, original_metrics, perm_results, OUTPUT_DIR)
        else:
            # Save evaluation results without permutation tests
            save_evaluation_results(rank, original_metrics, original_metrics, output_dir=OUTPUT_DIR)

        # --- 6. SAVE THE FACTOR MATRICES TO CSV ---
        print("  -> Saving factor matrices to CSV files...")
        
        # Create DataFrames for easier saving with headers
        agent_df = pd.DataFrame(
            agent_factors,
            columns=[f'LatentFactor_{i+1}' for i in range(rank)]
        )
        agent_df.index.name = 'AgentID'
        
        trial_df = pd.DataFrame(
            trial_factors,
            columns=[f'LatentFactor_{i+1}' for i in range(rank)]
        )
        trial_df.index.name = 'Trial'
        
        feature_df = pd.DataFrame(
            feature_factors,
            index=feature_names,  # Use feature names for the index
            columns=[f'LatentFactor_{i+1}' for i in range(rank)]
        )
        feature_df.index.name = 'Feature'
        
        # Define output paths
        agent_path = os.path.join(OUTPUT_DIR, f'agent_factors_rank_{rank}.csv')
        trial_path = os.path.join(OUTPUT_DIR, f'trial_factors_rank_{rank}.csv')
        feature_path = os.path.join(OUTPUT_DIR, f'feature_factors_rank_{rank}.csv')
        
        # Save to CSV
        agent_df.to_csv(agent_path)
        trial_df.to_csv(trial_path)
        feature_df.to_csv(feature_path)
        
        print(f"     Successfully saved files to '{OUTPUT_DIR}' directory.")

