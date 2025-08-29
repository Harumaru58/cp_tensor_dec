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
        
        # Enhanced tracking for behavioral analysis
        self.action_history = []
        self.reward_history = []
        self.consecutive_actions = 0
        self.last_action = None
        self.total_rewards = 0
        self.action_counts = np.zeros(n_actions)
        self.successful_pulls = np.zeros(n_actions)
        self.trial_count = 0
        
        # Advanced behavioral metrics
        self.action_switches = 0
        self.reward_streak = 0
        self.loss_streak = 0
        self.max_reward_streak = 0
        self.max_loss_streak = 0
        self.early_performance = 0  # First 10 trials
        self.mid_performance = 0    # Trials 10-50
        self.late_performance = 0   # Last 10 trials
        self.performance_trend = 0
        self.exploration_rate = 0
        self.exploitation_rate = 0

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
        
        # Update basic tracking
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.total_rewards += reward
        self.action_counts[action] += 1
        self.successful_pulls[action] += reward
        self.trial_count += 1
        
        # Track consecutive actions
        if self.last_action == action:
            self.consecutive_actions += 1
        else:
            self.consecutive_actions = 1
            self.action_switches += 1
        self.last_action = action
        
        # Track reward streaks
        if reward == 1:
            self.reward_streak += 1
            self.loss_streak = 0
            self.max_reward_streak = max(self.max_reward_streak, self.reward_streak)
        else:
            self.loss_streak += 1
            self.reward_streak = 0
            self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)
        
        # Update performance phases
        if self.trial_count == 10:
            self.early_performance = np.mean(self.reward_history[:10])
        elif self.trial_count == 50:
            self.mid_performance = np.mean(self.reward_history[10:50])
        elif self.trial_count >= 140:
            self.late_performance = np.mean(self.reward_history[-10:])
            if self.trial_count >= 50:
                self.performance_trend = self.late_performance - self.early_performance
        
        # Calculate exploration vs exploitation
        if self.trial_count >= 20:
            recent_actions = self.action_history[-20:]
            unique_actions = len(set(recent_actions))
            self.exploration_rate = unique_actions / 2.0  # Normalized to [0,1]
            self.exploitation_rate = 1 - self.exploration_rate
        
        return td_error

    def get_internal_states(self, td_error):
        return {
            'q_value_0': self.Q[0],
            'q_value_1': self.Q[1],
            'td_error': td_error,
        }
    
    def get_enhanced_observable_features(self):
        """Get sophisticated observable features for classification"""
        if self.trial_count == 0:
            return {}
        
        # Basic ratios
        action_0_ratio = self.action_counts[0] / max(1, self.trial_count)
        action_1_ratio = self.action_counts[1] / max(1, self.trial_count)
        reward_rate = self.total_rewards / max(1, self.trial_count)
        
        # Success rates per action
        action_0_success_rate = self.successful_pulls[0] / max(1, self.action_counts[0])
        action_1_success_rate = self.successful_pulls[1] / max(1, self.action_counts[1])
        
        # Behavioral consistency
        consecutive_action_ratio = self.consecutive_actions / max(1, self.trial_count)
        action_switch_rate = self.action_switches / max(1, self.trial_count - 1)
        action_consistency = 1 - action_switch_rate
        
        # Streak metrics
        reward_streak_ratio = self.reward_streak / max(1, self.trial_count)
        max_reward_streak_ratio = self.max_reward_streak / max(1, self.trial_count)
        max_loss_streak_ratio = self.max_loss_streak / max(1, self.trial_count)
        
        # Performance phases
        early_perf = self.early_performance if self.trial_count >= 10 else 0
        mid_perf = self.mid_performance if self.trial_count >= 50 else 0
        late_perf = self.late_performance if self.trial_count >= 140 else 0
        perf_trend = self.performance_trend if self.trial_count >= 140 else 0
        
        # Exploration metrics
        exploration_rate = self.exploration_rate if self.trial_count >= 20 else 0
        exploitation_rate = self.exploitation_rate if self.trial_count >= 20 else 0
        
        # Advanced behavioral patterns
        if self.trial_count >= 30:
            recent_rewards = self.reward_history[-30:]
            recent_actions = self.action_history[-30:]
            
            # Reward volatility
            reward_volatility = np.std(recent_rewards) if len(recent_rewards) > 1 else 0
            
            # Action-reward correlation
            if len(recent_actions) > 1:
                action_reward_corr = np.corrcoef(recent_actions, recent_rewards)[0, 1]
                action_reward_corr = 0 if np.isnan(action_reward_corr) else action_reward_corr
            else:
                action_reward_corr = 0
            
            # Learning curve analysis
            first_third = np.mean(recent_rewards[:10]) if len(recent_rewards) >= 10 else 0
            last_third = np.mean(recent_rewards[-10:]) if len(recent_rewards) >= 10 else 0
            learning_improvement = last_third - first_third
        else:
            reward_volatility = 0
            action_reward_corr = 0
            learning_improvement = 0
        
        return {
            # Basic metrics
            'action_0_ratio': action_0_ratio,
            'action_1_ratio': action_1_ratio,
            'reward_rate': reward_rate,
            'action_0_success_rate': action_0_success_rate,
            'action_1_success_rate': action_1_success_rate,
            
            # Behavioral consistency
            'consecutive_action_ratio': consecutive_action_ratio,
            'action_switch_rate': action_switch_rate,
            'action_consistency': action_consistency,
            
            # Streak analysis
            'reward_streak_ratio': reward_streak_ratio,
            'max_reward_streak_ratio': max_reward_streak_ratio,
            'max_loss_streak_ratio': max_loss_streak_ratio,
            
            # Performance phases
            'early_performance': early_perf,
            'mid_performance': mid_perf,
            'late_performance': late_perf,
            'performance_trend': perf_trend,
            
            # Exploration metrics
            'exploration_rate': exploration_rate,
            'exploitation_rate': exploitation_rate,
            
            # Advanced patterns
            'reward_volatility': reward_volatility,
            'action_reward_correlation': action_reward_corr,
            'learning_improvement': learning_improvement,
            
            # Summary stats
            'total_rewards': self.total_rewards,
            'trial_count': self.trial_count
        }

    def reset(self):
        self.Q = np.zeros(self.n_actions)
        self.action_history = []
        self.reward_history = []
        self.consecutive_actions = 0
        self.last_action = None
        self.total_rewards = 0
        self.action_counts = np.zeros(self.n_actions)
        self.successful_pulls = np.zeros(self.n_actions)
        self.trial_count = 0
        
        # Reset enhanced metrics
        self.action_switches = 0
        self.reward_streak = 0
        self.loss_streak = 0
        self.max_reward_streak = 0
        self.max_loss_streak = 0
        self.early_performance = 0
        self.mid_performance = 0
        self.late_performance = 0
        self.performance_trend = 0
        self.exploration_rate = 0
        self.exploitation_rate = 0

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
                
                # Always include enhanced observable features
                enhanced_features = agent.get_enhanced_observable_features()
                trial_data.update(enhanced_features)

                all_trials_data.append(trial_data)
        
        return pd.DataFrame(all_trials_data)

if __name__ == "__main__":
    print("Enhanced Bandit Experiment with Sophisticated Observable Features")
    print("=" * 70)
    
    train_experiment = PhenotypeExperiment(n_agents=1000, n_trials=150, random_seed=456)
    np.random.seed(456)
    train_df = train_experiment.run_experiment(include_internal_states=False)
    
    test_experiment = PhenotypeExperiment(n_agents=1000, n_trials=150, random_seed=434)
    np.random.seed(434)
    test_df = test_experiment.run_experiment(include_internal_states=False)
    
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save enhanced datasets
    train_df.to_csv(os.path.join(output_dir, "enhanced_dataset_train_1000.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "enhanced_dataset_test_1000.csv"), index=False)
    
    print(f"Enhanced datasets saved:")
    print(f"  Training: {len(train_df)} trials, {len(train_df.columns)} features")
    print(f"  Testing: {len(test_df)} trials, {len(test_df.columns)} features")
    print(f"  Features: {list(train_df.columns)}")
