import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import argparse

def load_train_test_datasets():
    """Load the separate train and test datasets"""
    train_path = 'datasets/enhanced_dataset_train_1000.csv'
    test_path = 'datasets/enhanced_dataset_test_1000.csv'
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Enhanced train dataset not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Enhanced test dataset not found at {train_path}")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, test_data

def prepare_natural_features(data):
    """
    Prepare natural enhanced behavioral features without artificial manipulation.
    Uses only the features that are naturally computed from behavioral data.
    """
    print("Preparing natural enhanced behavioral features...")
    
    # Group by agent to get per-agent data
    agent_data_list = []
    phenotypes = []
    
    for agent_id in data['agent_id'].unique():
        agent_data = data[data['agent_id'] == agent_id].copy()
        agent_data = agent_data.sort_values('trial')
        
        if len(agent_data) == 0:
            continue
            
        # Get phenotype
        phenotype = agent_data['phenotype'].iloc[0]
        phenotypes.append(phenotype)
        
        # Get the last trial data (most complete feature set)
        last_trial = agent_data.iloc[-1]
        
        # Define natural enhanced behavioral features (no data leakage)
        natural_features = [
            'action_0_ratio', 'action_1_ratio', 'reward_rate',
            'action_0_success_rate', 'action_1_success_rate',
            'consecutive_action_ratio', 'action_switch_rate', 'action_consistency',
            'exploration_rate', 'exploitation_rate'
        ]
        
        # Extract features
        features = {}
        for feature_name in natural_features:
            if feature_name in last_trial:
                features[feature_name] = last_trial[feature_name]
            else:
                features[feature_name] = 0.0
        
        agent_data_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(agent_data_list)
    
    print(f"Natural feature preparation completed:")
    print(f"  Features shape: {features_df.shape}")
    print(f"  Natural behavioral features: {len(features_df.columns)}")
    print(f"  Number of phenotypes: {len(set(phenotypes))}")
    print(f"  Feature columns: {list(features_df.columns)}")
    
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
    ax.set_title(f'Confusion Matrix - Natural Enhanced Features - Accuracy: {accuracy:.4f}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Phenotype', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Phenotype', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax).set_label('Count', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = 'classification_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'confusion_matrix_natural_enhanced.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    """Main function for natural enhanced behavioral classification"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Natural enhanced behavioral feature classification")
    parser.add_argument("--rf-seed", type=int, default=42, 
                       help="Random Forest seed (default: 42)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NATURAL ENHANCED BEHAVIORAL CLASSIFICATION")
    print("=" * 60)
    print("Uses only natural behavioral features without artificial manipulation")
    
    try:
        # Load separate train and test datasets
        print("\nLoading enhanced datasets...")
        train_data, test_data = load_train_test_datasets()
        
        # Prepare natural training features
        print(f"\n1. Preparing training features...")
        X_train, y_train = prepare_natural_features(train_data)
        
        # Prepare natural test features
        print(f"\n2. Preparing test features...")
        X_test, y_test = prepare_natural_features(test_data)
        
        print(f"\nFeature preparation completed:")
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
        
        print(f"\nNATURAL ENHANCED CLASSIFICATION COMPLETED SUCCESSFULLY!")
        print(f"Final Accuracy: {accuracy:.4f}")
        
        return accuracy
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
