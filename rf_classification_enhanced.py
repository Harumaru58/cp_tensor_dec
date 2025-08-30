import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import argparse
from tensorly.decomposition import non_negative_parafac

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

def create_tensor_from_data(data, n_agents, n_trials, n_features=None):
    """Create a 3D tensor from the enhanced dataset with multiple features"""
    # If n_features not specified, use all available behavioral features
    if n_features is None:
        # Define the behavioral features to use (excluding metadata)
        feature_columns = ['action', 'reward', 'action_0_ratio', 'action_1_ratio', 'reward_rate', 
                          'action_0_success_rate', 'action_1_success_rate',
                          'consecutive_action_ratio', 'action_switch_rate', 'action_consistency',
                          'reward_streak_ratio', 'early_performance', 'mid_performance',
                          'exploration_rate', 'exploitation_rate', 'reward_volatility',
                          'action_reward_correlation', 'learning_improvement']
        
        # Filter to only include features that exist in the dataset
        available_features = [col for col in feature_columns if col in data.columns]
        n_features = len(available_features)
        print(f"Using {n_features} features: {available_features}")
    else:
        # Use only the first n_features available
        feature_columns = ['action', 'reward', 'action_0_ratio', 'action_1_ratio', 'reward_rate', 
                          'action_0_success_rate', 'action_1_success_rate',
                          'consecutive_action_ratio', 'action_switch_rate', 'action_consistency',
                          'reward_streak_ratio', 'early_performance', 'mid_performance',
                          'exploration_rate', 'exploitation_rate', 'reward_volatility',
                          'action_reward_correlation', 'learning_improvement']
        available_features = feature_columns[:n_features]
    
    # Create tensor: (agents, trials, features)
    tensor = np.zeros((n_agents, n_trials, n_features))
    
    for agent_id in range(n_agents):
        agent_data = data[data['agent_id'] == agent_id]
        if len(agent_data) == n_trials:
            for feat_idx, feature_name in enumerate(available_features):
                if feature_name in agent_data.columns:
                    tensor[agent_id, :, feat_idx] = agent_data[feature_name].values
                else:
                    # Fill with zeros if feature is missing
                    tensor[agent_id, :, feat_idx] = 0.0
    
    return tensor

def perform_cp_decomposition(tensor, rank=3, n_iter_max=5000, random_state=42):
    """Perform non-negative CP decomposition"""
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
    """Normalize CP factors using L1 norm and adjust lambda weights accordingly"""
    weights = cp_decomposition[0].copy()
    factors = [f.copy() for f in cp_decomposition[1]]
    
    for idx, factor in enumerate(factors):
        l1_norms = np.sum(np.abs(factor), axis=0)
        l1_norms[l1_norms < min_norm] = 1.0
        factors[idx] = factor / l1_norms[np.newaxis, :]
        weights *= l1_norms
    
    return (weights, factors)

def reorder_components_by_weights(weights, factors):
    """Reorder components by decreasing weight magnitude"""
    sorted_indices = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_indices]
    sorted_factors = [factor[:, sorted_indices] for factor in factors]
    return sorted_weights, sorted_factors

def extract_agent_factors(factors, agent_dim=0):
    """Extract agent factor matrix from CP decomposition"""
    return factors[agent_dim]

def prepare_features_tensor_cp(data, rank=3, n_iter_max=5000, random_state=42):
    """Prepare agent features using CP decomposition on tensor data"""
    print(f"Preparing features using CP decomposition (rank {rank})...")
    
    # Get unique agents and trials
    n_agents = data['agent_id'].nunique()
    n_trials = data.groupby('agent_id').size().iloc[0]
    
    print(f"Creating tensor with shape: ({n_agents}, {n_trials}, multiple_features)")
    
    # Create tensor from data with all available behavioral features
    tensor = create_tensor_from_data(data, n_agents, n_trials, n_features=None)
    
    # Perform CP decomposition
    print(f"Performing CP decomposition with rank {rank}...")
    cp_decomp = perform_cp_decomposition(tensor, rank, n_iter_max, random_state)
    
    # L1 normalize factors
    print("L1 normalizing factors...")
    weights, factors = manual_l1_normalize(cp_decomp)
    
    # Reorder components by weights
    print("Reordering components by weights...")
    sorted_weights, sorted_factors = reorder_components_by_weights(weights, factors)
    
    # Extract agent factors (first dimension)
    print("Extracting agent factors...")
    agent_factors = extract_agent_factors(sorted_factors, agent_dim=0)
    
    # Get phenotypes for each agent
    phenotypes = []
    for agent_id in range(n_agents):
        agent_phenotype = data[data['agent_id'] == agent_id]['phenotype'].iloc[0]
        phenotypes.append(agent_phenotype)
    
    print(f"CP decomposition completed:")
    print(f"  Feature matrix shape: {agent_factors.shape}")
    print(f"  Number of phenotypes: {len(set(phenotypes))}")
    
    return agent_factors, np.array(phenotypes), sorted_weights, sorted_factors

def prepare_features_direct(data):
    """Prepare agent features directly from enhanced behavioral features"""
    print("Preparing features directly from enhanced behavioral data...")
    
    # Get unique agents
    n_agents = data['agent_id'].nunique()
    
    # Group by agent and get the last trial data (most complete features)
    agent_features = []
    phenotypes = []
    
    for agent_id in range(n_agents):
        agent_data = data[data['agent_id'] == agent_id]
        if len(agent_data) == 0:
            continue
            
        # Get the last trial data (most complete feature set)
        last_trial = agent_data.iloc[-1]
        
        # Extract only the behavioral features (exclude metadata)
        feature_columns = ['action_0_ratio', 'action_1_ratio', 'reward_rate', 
                          'action_0_success_rate', 'action_1_success_rate',
                          'consecutive_action_ratio', 'action_switch_rate', 
                          'action_consistency', 'reward_streak_ratio',
                          'early_performance', 'mid_performance', 
                          'exploration_rate', 'exploitation_rate',
                          'reward_volatility', 'action_reward_correlation', 
                          'learning_improvement']
        
        features = []
        for col in feature_columns:
            if col in last_trial:
                features.append(last_trial[col])
            else:
                features.append(0.0)  # Default value if feature missing
        
        agent_features.append(features)
        phenotypes.append(last_trial['phenotype'])
    
    # Convert to numpy array
    feature_matrix = np.array(agent_features)
    print(f"Direct features completed:")
    print(f"  Feature matrix shape: {feature_matrix.shape}")
    print(f"  Number of phenotypes: {len(set(phenotypes))}")
    
    return feature_matrix, np.array(phenotypes), feature_columns

def prepare_features_pca(data, n_components=3, random_state=42):
    """Prepare agent features using PCA on enhanced behavioral features"""
    print(f"Preparing features using PCA ({n_components} components)...")
    
    # Get unique agents
    n_agents = data['agent_id'].nunique()
    
    # Group by agent and get the last trial data (most complete features)
    agent_features = []
    phenotypes = []
    
    for agent_id in range(n_agents):
        agent_data = data[data['agent_id'] == agent_id]
        if len(agent_data) == 0:
            continue
            
        # Get the last trial data (most complete feature set)
        last_trial = agent_data.iloc[-1]
        
        # Extract only the behavioral features (exclude metadata)
        feature_columns = ['action_0_ratio', 'action_1_ratio', 'reward_rate', 
                          'action_0_success_rate', 'action_1_success_rate',
                          'consecutive_action_ratio', 'action_switch_rate', 
                          'action_consistency', 'reward_streak_ratio',
                          'early_performance', 'mid_performance', 
                          'exploration_rate', 'exploitation_rate',
                          'reward_volatility', 'action_reward_correlation', 
                          'learning_improvement']
        
        features = []
        for col in feature_columns:
            if col in last_trial:
                features.append(last_trial[col])
            else:
                features.append(0.0)  # Default value if feature missing
        
        agent_features.append(features)
        phenotypes.append(last_trial['phenotype'])
    
    # Convert to numpy array
    feature_matrix = np.array(agent_features)
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Feature columns: {feature_columns}")
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    features_pca = pca.fit_transform(feature_matrix)
    
    print(f"PCA features completed:")
    print(f"  PCA features shape: {features_pca.shape}")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    print(f"  Number of phenotypes: {len(set(phenotypes))}")
    
    return features_pca, np.array(phenotypes), pca, feature_columns

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

def plot_confusion_matrix(y_test, y_pred, le, accuracy, method_name):
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
    ax.set_title(f'Confusion Matrix - {method_name} - Accuracy: {accuracy:.4f}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Phenotype', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Phenotype', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax).set_label('Count', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = 'classification_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'confusion_matrix_{method_name.lower().replace(" ", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    """Main function for enhanced classification with multiple approaches"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced classification with multiple approaches")
    parser.add_argument("--method", type=str, default="tensor_cp", 
                       choices=["tensor_cp", "direct", "pca"],
                       help="Classification method: tensor_cp, direct, or pca (default: tensor_cp)")
    parser.add_argument("--rank", type=int, default=5, 
                       help="CP decomposition rank or PCA components (default: 5)")
    parser.add_argument("--rf-seed", type=int, default=42, 
                       help="Random Forest seed (default: 42)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENHANCED CLASSIFICATION WITH MULTIPLE APPROACHES")
    print("=" * 60)
    print(f"Method: {args.method.upper()}")
    print(f"Components/Rank: {args.rank}")
    
    try:
        # Load separate train and test datasets
        print("\nLoading enhanced datasets...")
        train_data, test_data = load_train_test_datasets()
        
        # Prepare features based on selected method
        if args.method == "tensor_cp":
            print(f"\n1. Preparing training features with CP decomposition (rank {args.rank})...")
            X_train, y_train, weights_train, factors_train = prepare_features_tensor_cp(
                train_data, args.rank, random_state=args.rf_seed
            )
            
            print(f"\n2. Preparing test features with CP decomposition (rank {args.rank})...")
            X_test, y_test, weights_test, factors_test = prepare_features_tensor_cp(
                test_data, args.rank, random_state=args.rf_seed
            )
            
            method_name = f"Tensor CP (Rank {args.rank})"
            
        elif args.method == "direct":
            print(f"\n1. Preparing training features directly...")
            X_train, y_train, feature_columns = prepare_features_direct(train_data)
            
            print(f"\n2. Preparing test features directly...")
            X_test, y_test, feature_columns = prepare_features_direct(test_data)
            
            method_name = "Direct Features"
            
        elif args.method == "pca":
            print(f"\n1. Preparing training features with PCA ({args.rank} components)...")
            X_train, y_train, pca_train, feature_columns = prepare_features_pca(
                train_data, args.rank, args.rf_seed
            )
            
            print(f"\n2. Preparing test features using fitted PCA...")
            # Extract test features and apply PCA transformation
            test_agent_features = []
            test_phenotypes = []
            
            for agent_id in range(test_data['agent_id'].nunique()):
                agent_data = test_data[test_data['agent_id'] == agent_id]
                if len(agent_data) == 0:
                    continue
                    
                last_trial = agent_data.iloc[-1]
                
                features = []
                for col in feature_columns:
                    if col in last_trial:
                        features.append(last_trial[col])
                    else:
                        features.append(0.0)
                
                test_agent_features.append(features)
                test_phenotypes.append(last_trial['phenotype'])
            
            test_feature_matrix = np.array(test_agent_features)
            X_test = pca_train.transform(test_feature_matrix)
            y_test = np.array(test_phenotypes)
            
            method_name = f"PCA ({args.rank} components)"
        
        print(f"\nFeature preparation completed:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        # Train model
        print(f"\n3. Training Random Forest classifier...")
        rf, le = train_random_forest(X_train, y_train, random_state=args.rf_seed)
        
        # Evaluate model
        print(f"\n4. Evaluating model on test data...")
        y_test_encoded = le.transform(y_test)
        accuracy, y_pred, y_pred_proba = evaluate_model(rf, le, X_test, y_test_encoded)
        
        # Plot results
        print(f"\n5. Generating confusion matrix plot...")
        plot_confusion_matrix(y_test_encoded, y_pred, le, accuracy, method_name)
        
        print(f"\n{method_name.upper()} CLASSIFICATION COMPLETED SUCCESSFULLY!")
        print(f"Final Accuracy: {accuracy:.4f}")
        
        return accuracy
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
