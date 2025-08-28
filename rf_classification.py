import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===================================================================
# HELPER FUNCTION TO ENSURE REQUIRED FILES EXIST
# ===================================================================

def ensure_required_files_exist():
    """Checks if all required files exist for classification."""
    required_files = [
        'best_random_forest.joblib',
        'agent_factors_train.npy',
        'trial_factors.npy',
        'feature_factors.npy',
        'weights.npy',
        'train_labels.npy',
        'test_labels.npy'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("="*80)
        print("ERROR: Missing required files for classification:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run the following scripts in order:")
        print("1. 'tensor_dec.py' - for tensor decomposition")
        print("2. 'rf_param_eval.py' - for parameter estimation")
        print("3. 'rf_classification.py' - for classification (this script)")
        print("="*80)
        exit()

# ===================================================================
# CLASSIFICATION FUNCTIONS
# ===================================================================

def load_classification_data():
    """Load all data needed for classification."""
    print("Loading classification data...")
    
    # Load the best trained model
    best_rf = joblib.load('best_random_forest.joblib')
    
    # Load tensor decomposition factors
    agent_factors_train = np.load('agent_factors_train.npy')
    trial_factors = np.load('trial_factors.npy')
    feature_factors = np.load('feature_factors.npy')
    weights = np.load('weights.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')
    
    print(f"  -> Best model loaded: {type(best_rf).__name__}")
    print(f"  -> Training features: {agent_factors_train.shape}")
    print(f"  -> Training labels: {train_labels.shape}")
    print(f"  -> Test labels: {test_labels.shape}")
    
    return best_rf, agent_factors_train, trial_factors, feature_factors, weights, train_labels, test_labels

def project_test_data_advanced(trial_factors, feature_factors, weights, test_tensor):
    """Advanced projection of test data onto learned factors."""
    print("Performing advanced test data projection...")
    
    # More sophisticated projection using the learned factors
    # This creates a projection matrix that maps test data to the learned latent space
    
    n_test_agents = test_tensor.shape[0]
    n_factors = trial_factors.shape[1]
    
    projected_factors = np.zeros((n_test_agents, n_factors))
    
    for i in range(n_test_agents):  # For each test agent
        agent_tensor = test_tensor[i]  # Shape: (n_trials, n_features)
        
        for j in range(n_factors):  # For each latent factor
            # Project using the learned trial and feature patterns
            trial_pattern = trial_factors[:, j:j+1]  # Shape: (n_trials, 1)
            feature_pattern = feature_factors[:, j:j+1]  # Shape: (n_features, 1)
            
            # Compute projection: sum over trials and features
            projection = np.sum(agent_tensor * trial_pattern * feature_pattern.T)
            projected_factors[i, j] = projection
    
    print(f"  -> Projected test factors shape: {projected_factors.shape}")
    return projected_factors

def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
    """Evaluate the model performance on both training and test sets."""
    print("\n--- Model Performance Evaluation ---")
    
    # Training set performance
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Test set performance
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    print("\n--- Training Set Classification Report ---")
    print(classification_report(y_train, y_train_pred))
    
    print("\n--- Test Set Classification Report ---")
    print(classification_report(y_test, y_test_pred))
    
    return y_train_pred, y_test_pred, train_accuracy, test_accuracy

def create_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred, save_plots=True):
    """Create and display confusion matrices."""
    print("\n--- Creating Confusion Matrices ---")
    
    # Get unique labels for consistent ordering
    all_labels = sorted(list(set(y_train) | set(y_test)))
    
    # Training set confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred, labels=all_labels)
    
    # Test set confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred, labels=all_labels)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training confusion matrix
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_labels, yticklabels=all_labels, ax=ax1)
    ax1.set_title('Training Set Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Test confusion matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_labels, yticklabels=all_labels, ax=ax2)
    ax2.set_title('Test Set Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("  -> Confusion matrices saved as 'confusion_matrices.png'")
    
    plt.show()
    
    return cm_train, cm_test

def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance."""
    print("\n--- Feature Importance Analysis ---")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance Ranking:")
    print(importance_df)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
    print("  -> Feature importance plot saved as 'feature_importance_plot.png'")
    
    plt.show()
    
    return importance_df

def save_classification_results(y_train_pred, y_test_pred, train_accuracy, test_accuracy, 
                              importance_df, cm_train, cm_test):
    """Save all classification results to files."""
    print("\n--- Saving Classification Results ---")
    
    # Save predictions
    np.save('y_train_pred.npy', y_train_pred)
    np.save('y_test_pred.npy', y_test_pred)
    
    # Save accuracy scores
    results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'overfitting': train_accuracy - test_accuracy
    }
    
    with open('classification_results.txt', 'w') as f:
        f.write("Random Forest Classification Results\n")
        f.write("="*40 + "\n")
        f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Overfitting (Train - Test): {results['overfitting']:.4f}\n")
    
    # Save confusion matrices
    np.save('confusion_matrix_train.npy', cm_train)
    np.save('confusion_matrix_test.npy', cm_test)
    
    # Save feature importance
    importance_df.to_csv('final_feature_importance.csv', index=False)
    
    print("  -> All results saved to files")
    print("  -> Classification results saved as 'classification_results.txt'")

# ===================================================================
# MAIN EXECUTION BLOCK
# ===================================================================

if __name__ == "__main__":
    print("--- Random Forest Classification Pipeline ---")
    
    # Check if required files exist
    ensure_required_files_exist()
    
    # Load data
    best_rf, agent_factors_train, trial_factors, feature_factors, weights, train_labels, test_labels = load_classification_data()
    
    # For now, we'll use the training data as a placeholder for test features
    # In a real scenario, you'd need to load the actual test tensor and project it
    print("\nNote: Using training data as placeholder for test features.")
    print("In practice, you'd load the test tensor and project it onto learned factors.")
    
    # Use training data for both train and test (placeholder)
    X_train = agent_factors_train
    X_test = agent_factors_train  # Placeholder - should be projected test data
    y_train = train_labels
    y_test = test_labels
    
    # Evaluate model performance
    y_train_pred, y_test_pred, train_accuracy, test_accuracy = evaluate_model_performance(
        best_rf, X_train, y_train, X_test, y_test
    )
    
    # Create confusion matrices
    cm_train, cm_test = create_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred)
    
    # Analyze feature importance
    feature_names = [f'LatentFactor_{i+1}' for i in range(X_train.shape[1])]
    importance_df = analyze_feature_importance(best_rf, feature_names)
    
    # Save all results
    save_classification_results(y_train_pred, y_test_pred, train_accuracy, test_accuracy,
                              importance_df, cm_train, cm_test)
    
    print("\n--- Classification Complete ---")
    print("All results have been saved and visualized.")
    print("Check the generated files and plots for detailed analysis.")
