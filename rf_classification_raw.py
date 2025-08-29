import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def load_train_test_datasets():
    """Load the separate train and test datasets"""
    train_path = 'datasets/raw_dataset_train_1000.csv'
    test_path = 'datasets/raw_dataset_test_1000.csv'
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    # get only the first 500 rows of the test data
    #test_data = test_data.head(500)
    
    return train_data, test_data



def prepare_sliding_window_data(data, window_size=100):
    """
    Prepare data using sliding window approach to avoid future information leakage.
    Only uses early trials (0 to window_size-1) to predict phenotype.
    """
    # Group by agent to get per-agent data
    agent_data_list = []
    phenotypes = []
    
    for agent_id in data['agent_id'].unique():
        agent_data = data[data['agent_id'] == agent_id].copy()
        agent_data = agent_data.sort_values('trial')
        
        if len(agent_data) < window_size:  # Skip agents with insufficient data
            continue
            
        # Get phenotype
        phenotype = agent_data['phenotype'].iloc[0]
        phenotypes.append(phenotype)
        
        # Get raw action and reward values from early trials only
        early_data = agent_data.head(window_size)
        
        # Create features for early trials only
        features = {}
        for i, (_, row) in enumerate(early_data.iterrows()):
            features[f'action_trial_{i}'] = row['action']
            features[f'reward_trial_{i}'] = row['reward']
        
        agent_data_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(agent_data_list)
    
    print(f"Sliding window data preparation completed (window size {window_size}):")
    print(f"  Features shape: {features_df.shape}")
    print(f"  Number of phenotypes: {len(set(phenotypes))}")
    print(f"  Feature columns: {list(features_df.columns)}")
    print(f"  Note: Only early trials used - no future information leakage")
    
    return features_df, np.array(phenotypes)

def train_random_forest(X_train, y_train, random_state=42):
    """Train Random Forest classifier"""
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    print(f"Training Random Forest classifier with {X_train.shape[0]} samples and {X_train.shape[1]} features")
    print(f"Training data: {X_train.iloc[0].shape}")
    print(f"Training labels: {X_train.iloc[:3]}")
    rf.fit(X_train, y_train_encoded)
    
    return rf, le

def evaluate_model(rf, le, X_test, y_test_encoded):
    """Evaluate the trained model"""
    
    # Make predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return accuracy, y_pred, y_pred_proba

def plot_confusion_matrix(y_test, y_pred, le, accuracy):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    cm = confusion_matrix(y_test, y_pred)
    desired_order = ['fast_learner', 'slow_learner', 'explorer', 'exploiter', 'random']
    
    # Reorder confusion matrix
    current_order = list(le.classes_)
    order_mapping = []
    for desired_phenotype in desired_order:
        if desired_phenotype in current_order:
            order_mapping.append(current_order.index(desired_phenotype))
    
    cm_reordered = cm[order_mapping][:, order_mapping]
    reordered_labels = [desired_order[i] for i in range(len(order_mapping))]
    
    # Create heatmap
    im = ax.imshow(cm_reordered, cmap='Blues', aspect='auto')
    
    # Add annotations
    for i in range(len(reordered_labels)):
        for j in range(len(reordered_labels)):
            ax.text(j, i, str(cm_reordered[i, j]),
                   ha="center", va="center", color="black", fontsize=12)
    
    # Set labels
    ax.set_xticks(range(len(reordered_labels)))
    ax.set_yticks(range(len(reordered_labels)))
    ax.set_xticklabels(reordered_labels, rotation=45, ha='right')
    ax.set_yticklabels(reordered_labels, rotation=0)
    ax.set_title(f'Confusion Matrix - Accuracy: {accuracy:.4f}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Phenotype', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Phenotype', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax).set_label('Count', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = 'classification_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'confusion_matrix_raw.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    """Main function for raw behavioral classification"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Raw behavioral classification without tensor decomposition")
    parser.add_argument("--rf-seed", type=int, default=42, help="Random Forest seed (default: 42)")

    parser.add_argument("--window-size", type=int, default=30, 
                       help="Window size for sliding window approach (default: 30)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RAW ACTION-REWARD CLASSIFICATION")
    print("=" * 60)
    print("Uses raw action and reward values from early trials")
    
    try:
        # Load separate train and test datasets
        print("Loading datasets...")
        train_data, test_data = load_train_test_datasets()
        
        # Prepare raw action and reward data from early trials
        print(f"\n1. Preparing training data from early {args.window_size} trials...")
        X_train, y_train = prepare_sliding_window_data(train_data, args.window_size)
        
        print(f"\n2. Preparing test data from early {args.window_size} trials...")
        X_test, y_test = prepare_sliding_window_data(test_data, args.window_size)
        
        print(f"\nFeature extraction completed:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        # Train model
        print("\n3. Training Random Forest classifier...")
        rf, le = train_random_forest(X_train, y_train, random_state=args.rf_seed)
        
        # Evaluate model
        print("\n4. Evaluating model on test data...")
        y_test_encoded = le.transform(y_test)
        accuracy, y_pred, y_pred_proba = evaluate_model(rf, le, X_test, y_test_encoded)
        
        # Plot results
        print("\n5. Generating confusion matrix plot...")
        plot_confusion_matrix(y_test_encoded, y_pred, le, accuracy)

        print("CLASSIFICATION COMPLETED SUCCESSFULLY!")

        
        return accuracy
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
