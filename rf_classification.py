import os
import sys
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
    test_data = test_data.head(500)
    
    return train_data, test_data

def create_tensor_from_data(data, n_agents, n_trials, n_features=2):
    """Create a 3D tensor from the dataset"""
    # Reshape data into tensor: (agents, trials, features)
    # Features: action (0/1) and reward (0/1)
    tensor = np.zeros((n_agents, n_trials, n_features))
    
    for agent_id in range(n_agents):
        agent_data = data[data['agent_id'] == agent_id]
        if len(agent_data) == n_trials:
            tensor[agent_id, :, 0] = agent_data['action'].values  # actions
            tensor[agent_id, :, 1] = agent_data['reward'].values  # rewards
    
    return tensor

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

# def prepare_agent_features(data, rank=3, n_iter_max=5000, random_state=42):
def prepare_agent_features(data, rank=3, n_iter_max=5000, random_state=42, weights_explicit=None):

    """Prepare agent features using CP decomposition"""
    print(f"Preparing features for dataset with shape: {data.shape}")
    
    # Get unique agents and trials
    n_agents = data['agent_id'].nunique()
    n_trials = data.groupby('agent_id').size().iloc[0]
    
    print(f"Creating tensor with shape: ({n_agents}, {n_trials}, 2)")
    
    # Create tensor from data
    tensor = create_tensor_from_data(data, n_agents, n_trials, n_features=2)
    
    # Perform CP decomposition
    print(f"Performing CP decomposition with rank {rank}...")
    cp_decomp = perform_cp_decomposition(tensor, rank, n_iter_max, random_state)
    
    # L1 normalize factors
    print("L1 normalizing factors...")
    weights, factors = manual_l1_normalize(cp_decomp)
    #print(f"Weights: {weights}")
    #print(f"weights shape: {weights.shape}")

    #if weights_explicit is not None:
    #    weights = weights_explicit

    # Reorder components by weights
    print("Reordering components by weights...")
    sorted_weights, sorted_factors = reorder_components_by_weights(weights, factors)
    
    # Extract agent factors (first dimension)
    print("Extracting agent factors...")
    agent_factors = extract_agent_factors(sorted_factors, agent_dim=0)
    #agent_factors = extract_agent_factors(factors, agent_dim=0)

    
    # Get phenotypes for each agent
    phenotypes = []
    for agent_id in range(n_agents):
        agent_phenotype = data[data['agent_id'] == agent_id]['phenotype'].iloc[0]
        phenotypes.append(agent_phenotype)
    
    print(f"Feature matrix shape: {agent_factors.shape}")
    print(f"Number of phenotypes: {len(set(phenotypes))}")
    
    return agent_factors, np.array(phenotypes)

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
    """Plot confusion matrix heatmap and save to file"""
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    
    # Define the desired phenotype order
    desired_order = ['fast_learner', 'slow_learner', 'explorer', 'exploiter', 'random']
    
    # Reorder confusion matrix to match desired order
    # Get the current order from the label encoder
    current_order = list(le.classes_)
    
    # Create mapping from current order to desired order
    order_mapping = []
    for desired_phenotype in desired_order:
        if desired_phenotype in current_order:
            order_mapping.append(current_order.index(desired_phenotype))
    
    # Reorder the confusion matrix
    cm_reordered = cm[order_mapping][:, order_mapping]
    
    # Get the reordered phenotype labels
    reordered_labels = [desired_order[i] for i in range(len(order_mapping))]
    
    # Create heatmap with counts clearly displayed
    im = ax.imshow(cm_reordered, cmap='Blues', aspect='auto')
    
    # Add count annotations to each cell
    for i in range(len(reordered_labels)):
        for j in range(len(reordered_labels)):
            text = ax.text(j, i, str(cm_reordered[i, j]),
                          ha="center", va="center", 
                          color="black", fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(range(len(reordered_labels)))
    ax.set_yticks(range(len(reordered_labels)))
    ax.set_xticklabels(reordered_labels, rotation=45, ha='right')
    ax.set_yticklabels(reordered_labels, rotation=0)
    
    # Set title and labels
    ax.set_title(f'Confusion Matrix - Accuracy: {accuracy:.4f}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Phenotype', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Phenotype', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=12, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = 'classification_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Raw dataset classification")
    parser.add_argument("--rank", type=int, default=3, help="CP decomposition rank (default: 3)")
    parser.add_argument("--rf-seed", type=int, default=42, help="Random Forest seed (default: 42)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RAW DATASET CLASSIFICATION WITH CP DECOMPOSITION")
    print("=" * 60)
    
    try:
        # Load separate train and test datasets
        print("Loading datasets...")
        train_data, test_data = load_train_test_datasets()
        
        # Prepare features for both datasets using CP decomposition
        print(f"\nPreparing training features with rank {args.rank}...")
        X_train, y_train = prepare_agent_features(train_data, rank=args.rank, random_state=args.rf_seed)
        
        print(f"\nPreparing test features with rank {args.rank}...")
        #weights = np.array([49688.17821651, 41862.07007872, 35370.41078917])
        #weights = np.array([1,1,1])
        #X_test, y_test = prepare_agent_features(test_data, rank=args.rank, random_state=args.rf_seed, weights_explicit=weights)
        X_test, y_test = prepare_agent_features(test_data, rank=args.rank, random_state=args.rf_seed)
        # Train model
        print("\nTraining Random Forest classifier...")
        rf, le = train_random_forest(X_train, y_train, random_state=args.rf_seed)
        
        # Ensure test labels are encoded with the same encoder
        #print(f"Test labels: {y_test}")
        y_test_encoded = le.transform(y_test)
        #print(f"Test labels encoded: {le.transform(y_test)}")
        
        # Evaluate model
        print("\nEvaluating model...")
        accuracy, y_pred, y_pred_proba = evaluate_model(rf, le, X_test, y_test_encoded)
        
        # Plot results
        print("\nGenerating confusion matrix plot...")
        plot_confusion_matrix(y_test_encoded, y_pred, le, accuracy)
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return accuracy
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
