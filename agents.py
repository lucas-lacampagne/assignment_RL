import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import itertools
from IPython.display import clear_output
import time

class GeneralAgent:
    def __init__(self):
        pass
        
    def evaluate_agent(self, env, num_tries=100, max_steps=1000, epsilon_greedy=False):
        steps = []
        total_scores = []
        
        for _ in tqdm(range(num_tries)):
            current_steps = 0
            state, _ = env.reset()
            done = False
            while not done and current_steps<max_steps:
                action = self.select_action(state, epsilon_greedy)
                state, reward, done,_, info = env.step(action)
                current_steps +=1
            steps.append(current_steps)
            total_scores.append(info['score'])
        return pd.DataFrame([steps, total_scores]).transpose().rename(columns={0:"steps",1: "total_scores"})

    def visualize_agent(self, env, max_steps=100, epsilon_greedy=False):
        state, _ = env.reset()
        done = False
        steps = 0
        
        try:
            while not done and steps < max_steps:
                action = self.select_action(state, epsilon_greedy)
                state, reward, done,_, info = env.step(action)
                
                clear_output(wait=True)
                print(env.render())
                print(f"Step: {steps} | Score: {info['score']}")
                
                time.sleep(0.03)
                steps += 1
        except KeyboardInterrupt:
            pass

class MonteCarloAgent(GeneralAgent):
    def __init__(self, epsilon=0.1, gamma=0.99):
        self.q_table = defaultdict(lambda: np.zeros(2)) # 2 actions: Idle or Flap
        self.returns_sum = defaultdict(lambda: np.zeros(2))
        self.returns_count = defaultdict(lambda: np.zeros(2))
        self.epsilon = epsilon
        self.gamma = gamma

    def select_action(self, state, epsilon_greedy=True):
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon and epsilon_greedy:
            return np.random.choice([0, 1])
        return np.argmax(self.q_table[state])

    def update(self, episode):
        """
        episode: list of (state, action, reward) tuples
        """
        G = 0
        visited_state_actions = set()
        
        # Iterating backwards through the episode
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            G = self.gamma * G + reward
            
            # Every-visit update
            self.returns_sum[state][action] += G
            self.returns_count[state][action] += 1
            self.q_table[state][action] = self.returns_sum[state][action] / self.returns_count[state][action]

    def train(self, env, n_episodes, epsilon_greedy=True):
        for i in tqdm(range(n_episodes)):
            state, _ = env.reset()
            episode_data = []
            done = False
            
            while not done:
                action = self.select_action(state, epsilon_greedy)
                next_state, reward, done, _, info = env.step(action)
                
                episode_data.append((state, action, reward))
                state = next_state
                
            self.update(episode_data)

        env.close()

class SarsaLambdaAgent(GeneralAgent):
    def __init__(self, alpha=0.1, gamma=0.99, lmbda=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(2))
        self.el_trace = defaultdict(lambda: np.zeros(2)) # Eligibility trace
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon

    def select_action(self, state, epsilon_greedy=True):
        if np.random.random() < self.epsilon and epsilon_greedy:
            return np.random.choice([0, 1])
        return np.argmax(self.q_table[state])

    def update(self, s, a, r, s_next, a_next):
        # TD error (delta)
        td_error = r + self.gamma * self.q_table[s_next][a_next] - self.q_table[s][a]
        
        # Accumulating trace
        self.el_trace[s][a] += 1
        
        # Update Q-table and decay traces for all seen states
        # In practice, only iterate over non-zero traces for efficiency
        for state in list(self.el_trace.keys()):
            for action in range(2):
                self.q_table[state][action] += self.alpha * td_error * self.el_trace[state][action]
                self.el_trace[state][action] *= self.gamma * self.lmbda

    def train(self, env, n_episodes, epsilon_greedy=True):
        for i in tqdm(range(n_episodes)):
            state, _ =  env.reset() 
            action = self.select_action(state, epsilon_greedy)
            self.reset_trace()
            done = False
            
            while not done:
                next_state, reward, done,_, info = env.step(action)
                
                next_action = self.select_action(next_state, epsilon_greedy)
                self.update(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

        env.close()

    def train_without_tqdm(self, env, n_episodes, epsilon_greedy=True):
        for i in range(n_episodes):
            state, _ =  env.reset() 
            action = self.select_action(state, epsilon_greedy)
            self.reset_trace()
            done = False
            
            while not done:
                next_state, reward, done,_, info = env.step(action)
                
                next_action = self.select_action(next_state, epsilon_greedy)
                self.update(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action

        env.close()
    
    def grid_search_sarsa(self, env, alpha_space, lambda_space, gamma_space, episodes=1000):
        results = []
        ags={}
        
        # Generate all combinations of parameters
        configurations = list(itertools.product(alpha_space, lambda_space, gamma_space))
        
        for alpha, lmbda, gamma in tqdm(configurations):
            
            # Initialize agent with current hyperparams
            agent = SarsaLambdaAgent(alpha=alpha, lmbda=lmbda, gamma=gamma)
            
            agent.train_without_tqdm(env, episodes)
            
            # Evaluation Phase
            df = agent.evaluate_agent(env, num_tries=4000, max_steps=1000)
            results.append({
                'alpha': alpha,
                'lambda': lmbda,
                'gamma': gamma,
                'avg_time_alive': df['steps'].mean(),
                'avg_score': df['total_scores'].mean()
            })
            ags[(alpha, lmbda, gamma)]=agent
            
        return pd.DataFrame(results), ags

                
    def reset_trace(self):
        self.el_trace.clear()