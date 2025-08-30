#!/usr/bin/env python3
"""
Test projection method with different datasets to see seed effects
"""

import os
import subprocess
import sys

def test_projection_with_dataset(train_seed, test_seed, rank=5):
    """Test projection with specific dataset seeds"""
    print(f"\n{'='*60}")
    print(f"TESTING PROJECTION WITH DATASETS")
    print(f"Train seed: {train_seed}, Test seed: {test_seed}, Rank: {rank}")
    print(f"{'='*60}")
    
    # Check if datasets exist
    train_file = f"datasets/enhanced_dataset_train_1000_seed_{train_seed}.csv"
    test_file = f"datasets/enhanced_dataset_test_1000_seed_{test_seed}.csv"
    
    if not os.path.exists(train_file):
        print(f"Training dataset not found: {train_file}")
        return None
    
    if not os.path.exists(test_file):
        print(f"Test dataset not found: {test_file}")
        return None
    
    # Temporarily modify the projection script to use these datasets
    backup_file = "rf_classification_proj_backup.py"
    
    # Create backup
    subprocess.run(["cp", "rf_classification_proj.py", backup_file])
    
    try:
        # Modify the dataset paths in the projection script
        with open("rf_classification_proj.py", "r") as f:
            content = f.read()
        
        # Replace dataset paths
        content = content.replace(
            "train_path = 'datasets/enhanced_dataset_train_1000.csv'",
            f"train_path = 'datasets/enhanced_dataset_train_1000_seed_{train_seed}.csv'"
        )
        content = content.replace(
            "test_path = 'datasets/enhanced_dataset_test_1000.csv'",
            f"test_path = 'datasets/enhanced_dataset_test_1000_seed_{test_seed}.csv'"
        )
        
        with open("rf_classification_proj.py", "w") as f:
            f.write(content)
        
        # Run the projection script
        print(f"Running projection with rank {rank}...")
        result = subprocess.run([
            "python", "rf_classification_proj.py", "--rank", str(rank)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Extract accuracy from output
            lines = result.stdout.split('\n')
            accuracy = None
            for line in lines:
                if "Test Accuracy:" in line:
                    accuracy = float(line.split(":")[1].strip())
                    break
            
            print(f"✅ SUCCESS: Accuracy = {accuracy:.4f}")
            return accuracy
        else:
            print(f"❌ FAILED: {result.stderr}")
            return None
            
    finally:
        # Restore original file
        subprocess.run(["cp", backup_file, "rf_classification_proj.py"])
        subprocess.run(["rm", backup_file])

def main():
    """Test multiple dataset combinations"""
    
    # Test combinations
    test_combinations = [
        (123, 321, 5),  # New seeds
        (456, 654, 5),  # New seeds
        (789, 987, 5),  # New seeds
        (101, 404, 5),  # New seeds
        (202, 505, 5),  # New seeds
        (456, 434, 5),  # Original seeds
    ]
    
    results = {}
    
    print("TESTING PROJECTION WITH MULTIPLE DATASETS")
    print("=" * 60)
    
    for train_seed, test_seed, rank in test_combinations:
        accuracy = test_projection_with_dataset(train_seed, test_seed, rank)
        if accuracy is not None:
            results[f"{train_seed}_{test_seed}"] = accuracy
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    
    for combo, acc in results.items():
        print(f"Seeds {combo}: {acc:.4f}")
    
    if results:
        avg_accuracy = sum(results.values()) / len(results)
        print(f"\nAverage accuracy: {avg_accuracy:.4f}")
        print(f"Best accuracy: {max(results.values()):.4f}")
        print(f"Worst accuracy: {min(results.values()):.4f}")
        print(f"Accuracy range: {max(results.values()) - min(results.values()):.4f}")

if __name__ == "__main__":
    main()
