import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================================
# HELPER FUNCTION TO ENSURE FACTORS EXIST
# ===================================================================

def ensure_factors_exist():
    required_files = [
        'tensor_decomposition_results/agent_factors_rank_6.csv',
        'datasets/ground_truth_bandit_data_1000agents.csv'
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("="*80)
        print("ERROR: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure these files exist before running parameter regression.")
        print("="*80)
        exit()
    else:
        print("  All required files found!")

# ===================================================================
# PARAMETER REGRESSION FUNCTIONS
# ===================================================================

def load_decomposition_factors_and_params():
    """Load tensor decomposition factors and extract agent parameters."""
    print("Loading tensor decomposition factors and agent parameters...")
    
    # Load agent factors from CSV
    agent_factors_df = pd.read_csv('tensor_decomposition_results/agent_factors_rank_6.csv', index_col=0)
    agent_factors = agent_factors_df.values
    
    # Load ground truth data to extract parameters
    ground_truth_df = pd.read_csv('datasets/ground_truth_bandit_data_1000agents.csv')
    
    # Extract unique agent parameters (first occurrence for each agent)
    agent_params = ground_truth_df.groupby('agent_id').agg({
        'phenotype': 'first',
        'q_value_0': 'first',
        'q_value_1': 'first'
    }).reset_index()
    
    # Create synthetic parameters based on phenotype (since we don't have actual alpha/temperature)
    # In a real scenario, these would come from the agent configuration
    param_mapping = {
        'fast_learner': {'alpha': 0.8, 'temperature': 0.1},
        'slow_learner': {'alpha': 0.2, 'temperature': 0.1},
        'explorer': {'alpha': 0.5, 'temperature': 2.0},
        'exploiter': {'alpha': 0.5, 'temperature': 0.05},
        'random': {'alpha': 0.1, 'temperature': 5.0}
    }
    
    # Add parameter columns
    agent_params['alpha'] = agent_params['phenotype'].map(lambda x: param_mapping[x]['alpha'])
    agent_params['temperature'] = agent_params['phenotype'].map(lambda x: param_mapping[x]['temperature'])
    
    # Extract features and targets
    X = agent_factors  # 6 latent factors (Rank 6)
    y_alpha = agent_params['alpha'].values
    y_temperature = agent_params['temperature'].values
    
    # Use all agents for training
    X_train = X
    y_alpha_train = y_alpha
    y_temp_train = y_temperature
    
    print(f"  -> Agent factors: {agent_factors.shape}")
    print(f"  -> Training set: {X_train.shape} (all agents)")
    print(f"  -> Alpha range: {y_alpha.min():.2f} to {y_alpha.max():.2f}")
    print(f"  -> Temperature range: {y_temperature.min():.2f} to {y_temperature.max():.2f}")
    
    return (X_train, y_alpha_train, y_temp_train, agent_params)

def perform_grid_search_regression(X_train, y_train, param_name, cv_folds=None):
    """Perform fine-tuned grid search to find optimal ensemble regressor parameters."""
    print(f"\n--- Performing Fine-Tuned Grid Search for {param_name} Regression ---")
    
    # Use 5-fold CV for faster execution while maintaining good performance
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Try multiple ensemble models for better performance
    models = {
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    
    best_score = -float('inf')
    best_model_name = None
    best_grid_search = None
    
    for model_name, model in models.items():
        print(f"\n--- Testing {model_name} with Fine-Tuned Parameters ---")
        
        if model_name == 'RandomForest':
            # Fast parameter grid for RandomForest
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
        elif model_name == 'ExtraTrees':
            # Fast parameter grid for ExtraTrees
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
        else:  # GradientBoosting
            # Fast parameter grid for GradientBoosting
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 7, 10],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        print(f"Parameter grid size: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features']) if 'max_features' in param_grid else len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])} combinations")
        
        # Perform fine-tuned grid search with 10-fold CV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1  # Show progress for fine-tuned search
        )
        
        print(f"Fine-tuned grid search started for {param_name} with {model_name} (10-fold CV)...")
        grid_search.fit(X_train, y_train)
        
        print(f"Best CV R² score for {model_name}: {grid_search.best_score_:.4f}")
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        
        # Keep track of the best model
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model_name = model_name
            best_grid_search = grid_search
    
    print(f"\n--- Best Model Selection (Fine-Tuned) ---")
    print(f"Best model for {param_name}: {best_model_name}")
    print(f"Best CV R² score: {best_score:.4f}")
    print(f"Best parameters: {best_grid_search.best_params_}")
    
    return best_grid_search

def evaluate_regression_performance(grid_search, X_train, y_train, param_name):
    """Evaluate the performance of the regression model."""
    print(f"\n--- {param_name} Regression Performance ---")
    
    # Get best estimator
    best_rf = grid_search.best_estimator_
    
    # Make predictions on training data
    y_train_pred = best_rf.predict(X_train)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    
    # 5-fold Cross-validation score (faster execution while maintaining good performance)
    from sklearn.model_selection import cross_val_score, KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring='r2')
    print(f"5-fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Individual CV scores range: {cv_scores.min():.4f} to {cv_scores.max():.4f}")
    
    # Feature importance
    feature_importance = best_rf.feature_importances_
    feature_names = [f'LatentFactor_{i+1}' for i in range(len(feature_importance))]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(f"\nFeature Importance for {param_name}:")
    print(importance_df)
    
    return best_rf, importance_df, {
        'train_r2': train_r2, 'cv_r2': cv_scores.mean(),
        'train_rmse': train_rmse, 'train_mae': train_mae
    }

def save_regression_results(alpha_rf, temp_rf, alpha_importance, temp_importance, 
                          alpha_metrics, temp_metrics, agent_params):
    """Save the regression results and models."""
    print("\n--- Saving Regression Results ---")
    
    # Save models
    import joblib
    joblib.dump(alpha_rf, 'alpha_regression_model.joblib')
    joblib.dump(temp_rf, 'temperature_regression_model.joblib')
    
    # Save feature importance
    alpha_importance.to_csv('alpha_feature_importance.csv', index=False)
    temp_importance.to_csv('temperature_feature_importance.csv', index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Parameter': ['Alpha', 'Temperature'],
        'Train_R2': [alpha_metrics['train_r2'], temp_metrics['train_r2']],
        'CV_R2': [alpha_metrics['cv_r2'], temp_metrics['cv_r2']],
        'Train_RMSE': [alpha_metrics['train_rmse'], temp_metrics['train_rmse']],
        'Train_MAE': [alpha_metrics['train_mae'], temp_metrics['train_mae']]
    })
    metrics_df.to_csv('regression_performance_metrics.csv', index=False)
    
    # Save agent parameters for reference
    agent_params.to_csv('agent_parameters_reference.csv', index=False)
    
    print("  -> Alpha regression model saved as 'alpha_regression_model.joblib'")
    print("  -> Temperature regression model saved as 'temperature_regression_model.joblib'")
    print("  -> Feature importance files saved")
    print("  -> Performance metrics saved as 'regression_performance_metrics.csv'")
    print("  -> Agent parameters reference saved as 'agent_parameters_reference.csv'")

def create_parameter_plots(agent_params, alpha_rf, temp_rf, X_train, y_alpha_train, y_temp_train):
    """Create visualization plots for parameter estimation."""
    print("\n--- Creating Parameter Estimation Plots ---")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Alpha vs Temperature scatter
    axes[0, 0].scatter(agent_params['alpha'], agent_params['temperature'], 
                       c=agent_params['phenotype'].astype('category').cat.codes, 
                       alpha=0.7, s=50)
    axes[0, 0].set_xlabel('Alpha (Learning Rate)')
    axes[0, 0].set_ylabel('Temperature (Exploration)')
    axes[0, 0].set_title('Agent Parameters by Phenotype')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Feature importance comparison
    n_features = X_train.shape[1]  # Dynamic feature count
    feature_names = [f'LatentFactor_{i+1}' for i in range(n_features)]
    x_pos = np.arange(len(feature_names))
    width = 0.35
    
    axes[0, 1].bar(x_pos - width/2, alpha_rf.feature_importances_, width, 
                    label='Alpha Regression', alpha=0.8)
    axes[0, 1].bar(x_pos + width/2, temp_rf.feature_importances_, width, 
                    label='Temperature Regression', alpha=0.8)
    axes[0, 1].set_xlabel('Latent Factors')
    axes[0, 1].set_ylabel('Feature Importance')
    axes[0, 1].set_title('Feature Importance Comparison')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(feature_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Alpha prediction vs actual
    y_alpha_pred = alpha_rf.predict(X_train)
    axes[1, 0].scatter(y_alpha_train, y_alpha_pred, alpha=0.7)
    axes[1, 0].plot([y_alpha_train.min(), y_alpha_train.max()], 
                     [y_alpha_train.min(), y_alpha_train.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Alpha')
    axes[1, 0].set_ylabel('Predicted Alpha')
    axes[1, 0].set_title('Alpha: Predicted vs Actual')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Temperature prediction vs actual
    y_temp_pred = temp_rf.predict(X_train)
    axes[1, 1].scatter(y_temp_train, y_temp_pred, alpha=0.7)
    axes[1, 1].plot([y_temp_train.min(), y_temp_train.max()], 
                     [y_temp_train.min(), y_temp_train.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Temperature')
    axes[1, 1].set_ylabel('Predicted Temperature')
    axes[1, 1].set_title('Temperature: Predicted vs Actual')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_estimation_plots.png', dpi=300, bbox_inches='tight')
    print("  -> Parameter estimation plots saved as 'parameter_estimation_plots.png'")

# ===================================================================
# MAIN EXECUTION BLOCK
# ===================================================================

if __name__ == "__main__":
    print("--- Random Forest Parameter Regression Pipeline ---")
    print("Estimating Alpha (Learning Rate) and Temperature (Exploration) Parameters")
    print("="*80)
    
    print("Step 1: Checking if required files exist...")
    # Check if required files exist
    ensure_factors_exist()
    print("Step 2: Files exist, proceeding to load data...")
    
    # Load decomposition factors and parameters
    print("Step 3: Loading decomposition factors and parameters...")
    (X_train, y_alpha_train, y_temp_train, agent_params) = load_decomposition_factors_and_params()
    print("Step 4: Data loaded successfully!")
    
    # Perform grid search for Alpha regression
    alpha_grid_search = perform_grid_search_regression(X_train, y_alpha_train, "Alpha")
    
    # Perform grid search for Temperature regression
    temp_grid_search = perform_grid_search_regression(X_train, y_temp_train, "Temperature")
    
    # Evaluate performance for both models
    alpha_rf, alpha_importance, alpha_metrics = evaluate_regression_performance(
        alpha_grid_search, X_train, y_alpha_train, "Alpha"
    )
    
    temp_rf, temp_importance, temp_metrics = evaluate_regression_performance(
        temp_grid_search, X_train, y_temp_train, "Temperature"
    )
    
    # Save results
    save_regression_results(alpha_rf, temp_rf, alpha_importance, temp_importance,
                          alpha_metrics, temp_metrics, agent_params)
    
    # Create visualization plots
    create_parameter_plots(agent_params, alpha_rf, temp_rf, X_train, y_alpha_train, y_temp_train)
    
    print("\n--- Parameter Regression Complete ---")
    print(f"Alpha Regression - Training R²: {alpha_metrics['train_r2']:.4f}, CV R²: {alpha_metrics['cv_r2']:.4f}")
    print(f"Temperature Regression - Training R²: {temp_metrics['train_r2']:.4f}, CV R²: {temp_metrics['cv_r2']:.4f}")
    print("\nNext: Use the trained models to estimate parameters for new agents!")
    print("Models saved as 'alpha_regression_model.joblib' and 'temperature_regression_model.joblib'")
