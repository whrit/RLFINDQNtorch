# env/trading_env.py

import gym
import numpy as np
from gym import spaces
import torch
import math


class TradingEnv(gym.Env):
    def __init__(self, data, action_name, gamma=0.95, n_step=5, batch_size=128, 
                 start_index_reward=0, transaction_cost=0.001, initial_balance=10000):
        super(TradingEnv, self).__init__()

        self.data = data
        self.states = self.data.values
        self.current_state_index = -1
        self.own_share = False
        self.batch_size = batch_size
        self.n_step = n_step
        self.position = 0
        self.initial_balance = initial_balance  # Changed from hardcoded 1000
        self.balance = initial_balance
        self.gamma = gamma
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.close_price = list(self.data['close'])
        self.action_name = action_name
        self.code_to_action = {0: 'buy', 1: 'hold', 2: 'sell'}  # Changed 'None' to 'hold'
        self.start_index_reward = start_index_reward
        self.trading_cost_ratio = transaction_cost
        
        # Track statistics
        self.total_trades = 0
        self.successful_trades = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # buy, hold, sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.states[0]),), dtype=np.float32)

        self.state_size = len(self.states[0])

    def reset(self):
        """Reset the environment to initial state"""
        self.current_state_index = -1
        self.own_share = False
        self.balance = self.initial_balance
        self.total_trades = 0
        self.successful_trades = 0
        return self.get_current_state()

    def step(self, action):
        """
        Execute one step in the environment
        Returns: done, reward, next_state, info
        """
        done = False
        next_state = None
        info = {}  # Added info dict for additional information

        if self.current_state_index + self.n_step < len(self.states):
            next_state = self.states[self.current_state_index + self.n_step]
        else:
            done = True

        # Track trades
        if action != 1:  # If not holding
            self.total_trades += 1

        if action == 0:  # Buy
            self.own_share = True
        elif action == 2:  # Sell
            self.own_share = False

        reward = 0
        if not done:
            reward = self.get_reward(action)
            if reward > 0:
                self.successful_trades += 1

        # Add additional info
        info = {
            'total_trades': self.total_trades,
            'success_rate': self.successful_trades / max(1, self.total_trades),
            'current_balance': self.balance,
            'position': 'long' if self.own_share else 'flat'
        }

        return done, reward, next_state, info

    def get_reward(self, action):
        """Calculate reward with transaction costs and n-step returns"""
        reward_index_first = self.current_state_index + self.start_index_reward
        reward_index_last = min(
            self.current_state_index + self.start_index_reward + self.n_step,
            len(self.close_price) - 1
        )

        p1 = self.close_price[reward_index_first]
        p2 = self.close_price[reward_index_last]

        reward = 0
        if p1 and p2 != 0:
            if action == 0 or (action == 1 and self.own_share):  # Buy or Hold Long
                # Apply transaction costs and calculate n-step return
                reward = ((1 - self.trading_cost_ratio) ** 2 * p2 / p1 - 1) * 100
                # Discount future rewards
                reward *= (self.gamma ** (reward_index_last - reward_index_first))
            
            elif action == 2 or (action == 1 and not self.own_share):  # Sell or Stay Flat
                reward = ((1 - self.trading_cost_ratio) ** 2 * p1 / p2 - 1) * 100
                reward *= (self.gamma ** (reward_index_last - reward_index_first))

        # Update balance (simplified)
        self.balance *= (1 + reward/100)
        return reward

    def __iter__(self):
        """Iterator for batch processing"""
        self.index_batch = 0
        self.num_batch = math.ceil(len(self.states) / self.batch_size)
        return self

    def __next__(self):
        if self.index_batch < self.num_batch:
            start_idx = self.index_batch * self.batch_size
            end_idx = min((self.index_batch + 1) * self.batch_size, len(self.states))
            batch = [torch.tensor([s], dtype=torch.float, device=self.device) 
                    for s in self.states[start_idx:end_idx]]
            self.index_batch += 1
            return torch.cat(batch)
        raise StopIteration

    def get_total_reward(self, action_list):
        """Calculate total reward for a list of actions"""
        total_reward = 0
        self.reset()  # Reset environment before calculating total reward
        
        for a in action_list:
            if a == 0:
                self.own_share = True
            elif a == 2:
                self.own_share = False
            self.current_state_index += 1
            total_reward += self.get_reward(a)

        return total_reward

    def make_investment(self, action_list):
        """Record actions in the dataset"""
        self.data[self.action_name] = 'hold'  # Changed from 'None'
        i = self.start_index_reward + 1
        for a in action_list:
            self.data[self.action_name][i] = self.code_to_action[a]
            i += 1