import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import argparse
from tensorly.decomposition import non_negative_parafac

def load_train_test_datasets():
    """Load the separate train and test datasets"""
    train_path = 'datasets/enhanced_dataset_train_1000_seed_789.csv'
    test_path = 'datasets/enhanced_dataset_test_1000_seed_987.csv'
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Enhanced train dataset not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Enhanced test dataset not found at {test_path}")
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, test_data

def create_tensor_from_data(data, n_features=2):
    """
    Create 3D tensor (n_agents, n_trials, n_features) and return agent_id order.
    Ensures correct ordering and raises if trial counts mismatch.
    """
    agent_ids = np.sort(data['agent_id'].unique())
    n_agents = len(agent_ids)
    n_trials = data.groupby('agent_id').size().iloc[0]

    tensor = np.zeros((n_agents, n_trials, n_features))

    for i, aid in enumerate(agent_ids):
        agent_data = data[data['agent_id'] == aid].sort_values(by='trial')  # sort if trial index available
        if len(agent_data) != n_trials:
            raise ValueError(f"Agent {aid} has {len(agent_data)} trials, expected {n_trials}.")
        tensor[i, :, 0] = agent_data['action'].values
        tensor[i, :, 1] = agent_data['reward'].values

    return tensor, agent_ids


def perform_cp_decomposition(tensor, rank=3, n_iter_max=5000, random_state=123):
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

def prepare_agent_features_cp(data, rank=3, n_iter_max=5000, random_state=456):
    """Prepare agent features using CP decomposition"""
    print(f"Preparing features for dataset with shape: {data.shape}")
    
    # Get unique agents and trials
    n_agents = data['agent_id'].nunique()
    n_trials = data.groupby('agent_id').size().iloc[0]
    
    print(f"Creating tensor with shape: ({n_agents}, {n_trials}, 2)")
    
    # Create tensor from data
    tensor, _ = create_tensor_from_data(data, n_features=2)
    
    # Perform CP decomposition
    print(f"Performing CP decomposition with rank {rank}...")
    cp_decomp = perform_cp_decomposition(tensor, rank, n_iter_max, random_state)
    
    # L1 normalize factors
    print("L1 normalizing factors...")
    weights, factors = manual_l1_normalize(cp_decomp)
    #print the first example of the factors
    print(f"First example of factors: {factors[0][0, :]}")
    print(f"First example of weights: {weights[0]}")
    print(f"Shape of factors: {factors[0].shape}")
    print(f"Shape of weights: {weights.shape}")
    # len of factors
    print(f"Length of factors: {len(factors)}")
    # len of weights
    print(f"Length of weights: {len(weights)}")
 
   
    
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
    
    print(f"Feature matrix shape: {agent_factors.shape}")
    print(f"Number of phenotypes: {len(set(phenotypes))}")
    
    return agent_factors, np.array(phenotypes), sorted_weights, sorted_factors

def project_test_data_onto_train_space(test_tensor, B_train, C_train, weights_train, A_test):
    """
    Project test data onto the training space using Khatri-Rao product and NNLS
    """
    n_agents, n_trials, n_features = test_tensor.shape
    rank = B_train.shape[1]
    test_tensor_2d = test_tensor.reshape(n_agents, -1)
    
    # Khatri-Rao: (n_trials * n_features, rank)
    kr_product = np.zeros((n_trials * n_features, rank))
    for r in range(rank):
        kr_product[:, r] = np.kron(B_train[:, r], C_train[:, r])

    
    # Fold weights so NNLS returns a_i (not λ a_i)
    kr_weighted = kr_product * weights_train[np.newaxis, :]

    from scipy.optimize import nnls
    projected_A = np.zeros((n_agents, rank))
    for i in range(n_agents):
        coeff, _ = nnls(kr_weighted, test_tensor_2d[i, :])
        projected_A[i, :] = coeff

    print(f"Khatri-Rao product shape: {kr_product.shape}")
    print(f"Projected A_test shape: {projected_A.shape}")
    return projected_A

def validate_projection(test_tensor, B_train, C_train, weights_train, A_test_projected):
    """
    Comprehensive validation of the projection results.
    Checks reconstruction error, numerical properties, and sanity tests.
    """
    
    # 1. Reconstruct test tensor using projected factors
    n_agents, n_trials, n_features = test_tensor.shape
    rank = A_test_projected.shape[1]
    
    # Reconstruct: A_test_projected @ (B_train ⊙ C_train)^T
    reconstructed = np.zeros_like(test_tensor)

    # Create Khatri-Rao product for reconstruction
    kr_product = np.zeros((n_trials * n_features, rank))
    for r in range(rank):
        kr_product[:, r] = np.kron(B_train[:, r], C_train[:, r])

    # Reconstruct using the projected A factors and Khatri-Rao product
    for i in range(n_agents):
        # A_test_projected[i, :] @ kr_product.T gives us the flattened reconstruction
        flattened = A_test_projected[i, :] @ kr_product.T
        reconstructed[i, :, :] = flattened.reshape(n_trials, n_features)

    # 2. Calculate reconstruction error
    mse = np.mean((test_tensor - reconstructed) ** 2)
    mae = np.mean(np.abs(test_tensor - reconstructed))
    
    print(f"Reconstruction Error:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # 3. Check numerical properties
    print(f"\nNumerical Properties:")
    print(f"  A_test_projected range: [{np.min(A_test_projected):.6f}, {np.max(A_test_projected):.6f}]")
    print(f"  A_test_projected mean: {np.mean(A_test_projected):.6f}")
    print(f"  A_test_projected std: {np.std(A_test_projected):.6f}")
    
    # 4. Sanity checks
    print(f"\nSanity Checks:")
    print(f"  All A_test_projected >= 0: {np.all(A_test_projected >= 0)}")
    print(f"  A_test_projected shape: {A_test_projected.shape}")
    print(f"  Test tensor shape: {test_tensor.shape}")
    
    return reconstructed, mse, mae

def train_random_forest(X_train, y_train, random_state=789):
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
    ax.set_title(f'Confusion Matrix - Accuracy: {accuracy:.4f}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Phenotype', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Phenotype', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax).set_label('Count', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = 'classification_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'confusion_matrix_proj.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {output_path}")
    
    plt.show()

def main():
    """Main function implementing proper projection approach"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Projection-based classification with CP decomposition")
    parser.add_argument("--rank", type=int, default=3, 
                       help="CP decomposition rank (default: 3)")
    parser.add_argument("--rf-seed", type=int, default=789, 
                       help="Random Forest seed (default: 789)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PROJECTION-BASED CLASSIFICATION WITH CP DECOMPOSITION")
    print("=" * 60)
    print("Uses CP decomposition, Khatri-Rao product, and NNLS projection")
    
    try:
        # Load separate train and test datasets
        print("Loading datasets...")
        train_data, test_data = load_train_test_datasets()
        
        # 1. Prepare training features with CP decomposition
        print(f"\n1. Preparing training features with rank {args.rank}...")
        A_train, y_train, weights_train, factors_train = prepare_agent_features_cp(
            train_data, args.rank, random_state=args.rf_seed
        )
        
        # Extract B and C factors from training
        B_train = factors_train[1]  # Trial factors
        C_train = factors_train[2]  # Feature factors
        
        print(f"Training factors extracted:")
        print(f"  A_train shape: {A_train.shape}")
        print(f"  B_train shape: {B_train.shape}")
        print(f"  C_train shape: {C_train.shape}")
        print(f"  Weights: {weights_train}")
        
        # 2. Create test tensor
        print(f"\n2. Creating test tensor...")
        n_agents_test = test_data['agent_id'].nunique()
        n_trials_test = test_data.groupby('agent_id').size().iloc[0]
        
        test_tensor, agent_ids = create_tensor_from_data(test_data, n_features=2)
        phenotypes = [test_data[test_data['agent_id']==aid]['phenotype'].iloc[0] for aid in agent_ids]
        print(f"Test tensor shape: {test_tensor.shape}")
        print(f"Number of phenotypes: {len(set(phenotypes))}")

        
        # 3. Project test data using Khatri-Rao product and NNLS
        print(f"\n3. Projecting test data using Khatri-Rao product and NNLS...")
        A_test_projected = project_test_data_onto_train_space(
            test_tensor, B_train, C_train, weights_train, A_train
        )
        
        # 4. Validate projection
        print(f"\n4. Validating projection results...")
        reconstructed, mse, mae = validate_projection(
            test_tensor, B_train, C_train, weights_train, A_test_projected
        )
        
        # 5. Get test phenotypes
        y_test = []
        for agent_id in range(n_agents_test):
            agent_phenotype = test_data[test_data['agent_id'] == agent_id]['phenotype'].iloc[0]
            y_test.append(agent_phenotype)
        y_test = np.array(y_test)
        
        print(f"\nTest data prepared:")
        print(f"  X_test shape: {A_test_projected.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        # 6. Train model
        print("\n5. Training Random Forest classifier...")
        rf, le = train_random_forest(A_train, y_train, random_state=args.rf_seed)
        
        # 7. Evaluate model on projected test data
        print("\n6. Evaluating model on projected test data...")
        y_test_encoded = le.transform(y_test)
        accuracy, y_pred, y_pred_proba = evaluate_model(rf, le, A_test_projected, y_test_encoded)
        
        # 8. Plot results
        print("\n7. Generating confusion matrix plot...")
        plot_confusion_matrix(y_test_encoded, y_pred, le, accuracy)
        
        print("PROJECTION-BASED CLASSIFICATION COMPLETED SUCCESSFULLY!")

        #print the shape of the test tensor and the train tensor
        print(f"Test tensor shape: {test_tensor.shape}")
        print(f"Train tensor shape: {A_train.shape}")

        #print the first example of the test tensor and the train tensor
        print(f"First example of test tensor: {test_tensor[0, :, :]}")
        print(f"First example of train tensor: {A_train[0, :]}")
        print(f"B_train shape: {B_train.shape}")
        print(f"C_train shape: {C_train.shape}")

        return accuracy
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
