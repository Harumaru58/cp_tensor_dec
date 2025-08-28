import numpy as np
import pandas as pd
import os

# ===================================================================
# PART 1: CORE COMPONENTS (AGENT AND ENVIRONMENT)
# ===================================================================

class TwoArmedBandit:
    """
    A simple Two-Armed Bandit environment.
    One arm has a higher probability of reward than the other.
    """
    def __init__(self, p=[0.7, 0.3]):
        """
        Initializes the bandit with reward probabilities for each arm.
        Args:
            p (list): A list of two floats, where p[0] is the reward probability
                      for arm 0 and p[1] is for arm 1.
        """
        if len(p) != 2:
            raise ValueError("Reward probability list 'p' must contain exactly two values.")
        self.p = np.array(p)

    def pull(self, action):
        """
        Pull one of the bandit's arms.
        Args:
            action (int): The arm to pull (0 or 1).
        Returns:
            int: The reward (1 for a win, 0 for a loss).
        """
        return np.random.binomial(1, self.p[action])

class QLearningAgent:
    """
    A Q-Learning agent that learns to solve the Two-Armed Bandit problem.
    It uses a softmax policy for action selection.
    """
    def __init__(self, alpha, temperature, n_actions=2):
        """
        Initializes the Q-Learning agent.
        Args:
            alpha (float): The learning rate (0 to 1).
            temperature (float): The exploration parameter for the softmax policy.
                                 Higher values lead to more random actions.
            n_actions (int): The number of possible actions.
        """
        self.alpha = alpha
        self.temperature = temperature
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)

    def choose_action(self):
        """
        Chooses an action using a softmax (Boltzmann) exploration strategy.
        Includes a robust safety check for temperature = 0.
        """
        # Robust safety check for greedy action selection
        if self.temperature == 0:
            # If multiple actions have the same max Q-value, choose randomly among them
            best_actions = np.where(self.Q == np.max(self.Q))[0]
            return np.random.choice(best_actions)
        
        # Standard softmax calculation for temp > 0
        # Add a stability term to prevent overflow with large Q-values
        q_stabilized = self.Q - np.max(self.Q)
        exp_q = np.exp(q_stabilized / self.temperature)
        probs = exp_q / np.sum(exp_q)
        
        return np.random.choice(self.n_actions, p=probs)

    def update(self, action, reward):
        """
        Updates the Q-value for the chosen action using the Q-learning rule.
        Args:
            action (int): The action that was taken.
            reward (int): The reward that was received.
        Returns:
            float: The temporal difference (TD) error for this update step.
        """
        old_q_value = self.Q[action]
        td_error = reward - old_q_value
        self.Q[action] = old_q_value + self.alpha * td_error
        return td_error

    def get_internal_states(self, td_error):
        """
        Returns a dictionary of the agent's current internal states for logging.
        Args:
            td_error (float): The TD error calculated during the last update.
        Returns:
            dict: A dictionary containing the agent's internal metrics.
        """
        return {
            'q_value_0': self.Q[0],
            'q_value_1': self.Q[1],
            'td_error': td_error,
        }

    def reset(self):
        """Resets the agent's Q-values to their initial state."""
        self.Q = np.zeros(self.n_actions)

# ===================================================================
# PART 2: THE EXPERIMENT ORCHESTRATOR
# ===================================================================

class PhenotypeExperiment:
    def __init__(self, n_agents=1000, n_trials=150, random_seed=42):
        self.n_agents = n_agents
        self.n_trials = n_trials
        self.phenotypes = {
            'fast_learner': {'alpha': 0.8, 'temperature': 0.1},
            'slow_learner': {'alpha': 0.2, 'temperature': 0.1},
            'explorer': {'alpha': 0.5, 'temperature': 1.0},
            'exploiter': {'alpha': 0.5, 'temperature': 0.05},
            'random': {'alpha': 0.1, 'temperature': 5.0}
        }
        if random_seed is not None:
            np.random.seed(random_seed)

    def _sample_phenotypes(self):
        """Assigns a phenotype to each agent, balanced across groups."""
        phenotype_keys = list(self.phenotypes.keys())
        # Repeat the list of phenotypes to cover all agents
        n_repeats = int(np.ceil(self.n_agents / len(phenotype_keys)))
        phenotype_list = np.repeat(phenotype_keys, n_repeats)[:self.n_agents]
        np.random.shuffle(phenotype_list)
        return phenotype_list

    def run_experiment(self, include_internal_states=False):
        """
        Runs the full simulation.
        Args:
            include_internal_states (bool): If True, the output dataframe will
                                            include the agent's Q-values and
                                            TD-error on each trial.
        Returns:
            pd.DataFrame: A dataframe containing the trial-by-trial data.
        """
        agent_phenotypes = self._sample_phenotypes()
        all_trials_data = []

        for agent_id, phenotype in enumerate(agent_phenotypes):
            params = self.phenotypes[phenotype]
            agent = QLearningAgent(alpha=params['alpha'], temperature=params['temperature'])
            bandit = TwoArmedBandit()

            for trial in range(self.n_trials):
                # 1. Agent chooses an action
                action = agent.choose_action()

                # 2. Environment gives a reward
                reward = bandit.pull(action)

                # 3. Agent updates its internal values
                td_error = agent.update(action, reward)

                # 4. Log the data for this trial
                trial_data = {
                    'agent_id': agent_id,
                    'trial': trial,
                    'phenotype': phenotype,
                    'action': action,
                    'reward': reward
                }

                # 5. Conditionally add internal states if requested
                if include_internal_states:
                    internal_states = agent.get_internal_states(td_error)
                    trial_data.update(internal_states)

                all_trials_data.append(trial_data)
        
        return pd.DataFrame(all_trials_data)

# ===================================================================
# PART 3: MAIN EXECUTION BLOCK
# ===================================================================

if __name__ == "__main__":
    # Create an instance of the experiment with 1000 agents for better results
    experiment = PhenotypeExperiment(n_agents=1000, n_trials=150)
    
    # --- Generate Dataset 1: The "Observable" Dataset ---
    # This dataset is what you'd use for the real test. It contains only
    # what an external observer could see: actions and rewards.
    print("Generating the 'Observable Only' dataset...")
    observable_df = experiment.run_experiment(include_internal_states=False)
    
    # Save the dataset
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    observable_df.to_csv(os.path.join(output_dir, f"observable_bandit_data_{experiment.n_agents}agents.csv"), index=False)
    
    print(f"Saved 'Observable Only' dataset with shape {observable_df.shape}")
    print("Head of the dataset:")
    print(observable_df.head())
    print("-" * 50)
    
    # --- Generate Dataset 2: The "Ground Truth" Dataset ---
    # This dataset includes the agent's internal "thoughts" (Q-values, TD-error).
    # It's useful for model validation, analysis, or as an easier training target.
    print("Generating the 'Ground Truth' dataset with internal states...")
    ground_truth_df = experiment.run_experiment(include_internal_states=True)
    
    # Save the dataset
    ground_truth_df.to_csv(os.path.join(output_dir, f"ground_truth_bandit_data_{experiment.n_agents}agents.csv"), index=False)

    print(f"Saved 'Ground Truth' dataset with shape {ground_truth_df.shape}")
    print("Head of the dataset:")
    print(ground_truth_df.head())
    print("-" * 50)
    
    print("Experiment complete. You now have two datasets in the 'datasets' folder.")
