import numpy as np
import pandas as pd
import os

class TwoArmedBandit:
    def __init__(self, p=[0.7, 0.3]):
        if len(p) != 2:
            raise ValueError("Reward probability list 'p' must contain exactly two values.")
        self.p = np.array(p)

    def pull(self, action):
        return np.random.binomial(1, self.p[action])

class QLearningAgent:
    def __init__(self, alpha, temperature, n_actions=2):
        self.alpha = alpha
        self.temperature = temperature
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)

    def choose_action(self):
        if self.temperature == 0:
            best_actions = np.where(self.Q == np.max(self.Q))[0]
            return np.random.choice(best_actions)
        
        q_stabilized = self.Q - np.max(self.Q)
        exp_q = np.exp(q_stabilized / self.temperature)
        probs = exp_q / np.sum(exp_q)
        
        return np.random.choice(self.n_actions, p=probs)

    def update(self, action, reward):
        old_q_value = self.Q[action]
        td_error = reward - old_q_value
        self.Q[action] = old_q_value + self.alpha * td_error
        return td_error

    def get_internal_states(self, td_error):
        return {
            'q_value_0': self.Q[0],
            'q_value_1': self.Q[1],
            'td_error': td_error,
        }

    def reset(self):
        self.Q = np.zeros(self.n_actions)

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
        phenotype_keys = list(self.phenotypes.keys())
        n_repeats = int(np.ceil(self.n_agents / len(phenotype_keys)))
        phenotype_list = np.repeat(phenotype_keys, n_repeats)[:self.n_agents]
        np.random.shuffle(phenotype_list)
        return phenotype_list

    def run_experiment(self, include_internal_states=False):
        agent_phenotypes = self._sample_phenotypes()
        all_trials_data = []

        for agent_id, phenotype in enumerate(agent_phenotypes):
            params = self.phenotypes[phenotype]
            agent = QLearningAgent(alpha=params['alpha'], temperature=params['temperature'])
            bandit = TwoArmedBandit()

            for trial in range(self.n_trials):
                action = agent.choose_action()
                reward = bandit.pull(action)
                td_error = agent.update(action, reward)

                trial_data = {
                    'agent_id': agent_id,
                    'trial': trial,
                    'phenotype': phenotype,
                    'action': action,
                    'reward': reward
                }

                if include_internal_states:
                    internal_states = agent.get_internal_states(td_error)
                    trial_data.update(internal_states)

                all_trials_data.append(trial_data)
        
        return pd.DataFrame(all_trials_data)

if __name__ == "__main__":
    train_experiment = PhenotypeExperiment(n_agents=1000, n_trials=150)
    train_df = train_experiment.run_experiment(include_internal_states=False)
    
    test_experiment = PhenotypeExperiment(n_agents=1000, n_trials=150)
    test_df = test_experiment.run_experiment(include_internal_states=False)
    
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "raw_dataset_train_1000.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "raw_dataset_test_1000.csv"), index=False)
