#!/usr/bin/env python3
"""
Generate multiple enhanced datasets with different random seeds
"""

import numpy as np
import pandas as pd
import os
from bandit_enhanced import PhenotypeExperiment

def generate_dataset_with_seed(seed, n_agents=1000, n_trials=150, dataset_type="train"):
    """Generate a dataset with a specific random seed"""
    print(f"Generating {dataset_type} dataset with seed {seed}...")
    
    # Set the random seed
    np.random.seed(seed)
    
    # Create experiment
    experiment = PhenotypeExperiment(n_agents=n_agents, n_trials=n_trials, random_seed=seed)
    
    # Run experiment
    df = experiment.run_experiment(include_internal_states=False)
    
    return df

def main():
    """Generate multiple datasets with different seeds"""
    
    # Define different seeds for variety
    seeds = {
        "train": [123, 456, 789, 101, 202],
        "test": [321, 654, 987, 404, 505]
    }
    
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING MULTIPLE ENHANCED DATASETS")
    print("=" * 60)
    
    # Generate training datasets
    print("\nGenerating training datasets...")
    for i, seed in enumerate(seeds["train"]):
        train_df = generate_dataset_with_seed(seed, dataset_type="train")
        filename = f"enhanced_dataset_train_1000_seed_{seed}.csv"
        filepath = os.path.join(output_dir, filename)
        train_df.to_csv(filepath, index=False)
        print(f"  Saved: {filename} ({len(train_df)} trials, {len(train_df.columns)} features)")
    
    # Generate test datasets
    print("\nGenerating test datasets...")
    for i, seed in enumerate(seeds["test"]):
        test_df = generate_dataset_with_seed(seed, dataset_type="test")
        filename = f"enhanced_dataset_test_1000_seed_{seed}.csv"
        filepath = os.path.join(output_dir, filename)
        test_df.to_csv(filepath, index=False)
        print(f"  Saved: {filename} ({len(test_df)} trials, {len(test_df.columns)} features)")
    
    # Also generate the original seed combinations for comparison
    print("\nGenerating original seed combinations...")
    
    # Original training seed (456)
    train_df_orig = generate_dataset_with_seed(456, dataset_type="train")
    train_df_orig.to_csv(os.path.join(output_dir, "enhanced_dataset_train_1000.csv"), index=False)
    print(f"  Saved: enhanced_dataset_train_1000.csv (original seed 456)")
    
    # Original test seed (434)
    test_df_orig = generate_dataset_with_seed(434, dataset_type="test")
    test_df_orig.to_csv(os.path.join(output_dir, "enhanced_dataset_test_1000.csv"), index=False)
    print(f"  Saved: enhanced_dataset_test_1000.csv (original seed 434)")
    
    print(f"\nAll datasets generated successfully!")
    print(f"Total datasets created: {len(seeds['train']) + len(seeds['test']) + 2}")
    print(f"Output directory: {output_dir}")
    
    # Show feature information
    print(f"\nFeature information:")
    print(f"  Number of features: {len(train_df_orig.columns)}")
    print(f"  Feature columns: {list(train_df_orig.columns)}")

if __name__ == "__main__":
    main()
