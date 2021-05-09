# Environments
import gym
from gym import spaces

# Computations
import pandas as pd
import numpy as np
import random

# RL Algorithms
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_checker import check_env


# Our custom learning environment
class CryptoEnv(gym.Env):
    # The 'human' render mode doesn't return anything.
    metadata = {'render.modes': ['human']}

    # Instantiate the environment
    def __init__(self, dfs, initial_balance=1000):
        super(CryptoEnv, self).__init__()
        # Define parameters
        self.dfs = dfs
        self.df_total_steps = len(self.dfs[0]) - 1
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.start_step = 0
        self.end_step = self.df_total_steps
        self.current_step = self.start_step
        # Define action space { 0: Hold, 1: Buy BTC, 2: Buy ETH, 3: Sell }
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        # Start with balance fully liquidated
        self.current_coin = 3
        # Define observation space
        self.observation_space = spaces.Box(low=0, high=3, shape=(10,), dtype=np.float32)

    # Reset the environment
    def reset(self, env_steps_size=0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.current_coin = 3
        self.start_step = 0
        self.end_step = self.df_total_steps
        self.current_step = self.start_step

        market_history_c1 = [self.dfs[0].loc[self.current_step, 'open'] / 100000000.0,
                             self.dfs[0].loc[self.current_step, 'high'] / 100000000.0,
                             self.dfs[0].loc[self.current_step, 'low'] / 100000000.0,
                             self.dfs[0].loc[self.current_step, 'close'] / 100000000.0,
                             self.dfs[0].loc[self.current_step, 'Volume USDT'] / 100000000.0,
                             self.dfs[1].loc[self.current_step, 'open'] / 100000000.0,
                             self.dfs[1].loc[self.current_step, 'high'] / 100000000.0,
                             self.dfs[1].loc[self.current_step, 'low'] / 100000000.0,
                             self.dfs[1].loc[self.current_step, 'close'] / 100000000.0,
                             self.dfs[1].loc[self.current_step, 'Volume USDT'] / 100000000.0]

        return np.array(market_history_c1).astype(np.float32)

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        price_of_btc = random.uniform(self.dfs[0].loc[self.current_step, 'open'],
                                      self.dfs[0].loc[self.current_step, 'close'])
        price_of_eth = random.uniform(self.dfs[1].loc[self.current_step, 'open'],
                                      self.dfs[1].loc[self.current_step, 'close'])

        held_coin_price = 0

        if self.current_coin == 0:
            held_coin_price = price_of_btc

        if self.current_coin == 1:
            held_coin_price = price_of_eth

        if action == 0:  # Hold
            pass

        elif action == 1 and self.balance > 0:
            # Sell all crypto and buy BCT with 100% of current balance
            self.current_coin = 0
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * held_coin_price
            self.crypto_held -= self.crypto_sold

            self.current_coin = 0
            self.crypto_bought = self.balance / price_of_btc
            self.balance -= self.crypto_bought * price_of_btc
            self.crypto_held += self.crypto_bought

        elif action == 2 and self.balance > 0:
            # Sell all crypto and buy ETC with 100% of current balance
            self.current_coin = 0
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * held_coin_price
            self.crypto_held -= self.crypto_sold

            self.current_coin = 1
            self.crypto_bought = self.balance / price_of_eth
            self.balance -= self.crypto_bought * price_of_eth
            self.crypto_held += self.crypto_bought

        elif action == 3 and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.current_coin = 0
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * held_coin_price
            self.crypto_held -= self.crypto_sold

        if self.current_coin == 0:
            held_coin_price = price_of_btc

        if self.current_coin == 1:
            held_coin_price = price_of_eth

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * held_coin_price

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        market_history_c1 = [self.dfs[0].loc[self.current_step, 'open'] / 100000000.0,
                             self.dfs[0].loc[self.current_step, 'high'] / 100000000.0,
                             self.dfs[0].loc[self.current_step, 'low'] / 100000000.0,
                             self.dfs[0].loc[self.current_step, 'close'] / 100000000.0,
                             self.dfs[0].loc[self.current_step, 'Volume USDT'] / 100000000.0,
                             self.dfs[1].loc[self.current_step, 'open'] / 100000000.0,
                             self.dfs[1].loc[self.current_step, 'high'] / 100000000.0,
                             self.dfs[1].loc[self.current_step, 'low'] / 100000000.0,
                             self.dfs[1].loc[self.current_step, 'close'] / 100000000.0,
                             self.dfs[1].loc[self.current_step, 'Volume USDT'] / 100000000.0]

        obs = np.array(market_history_c1).astype(np.float32)

        return obs, reward, self.current_step == self.df_total_steps, {}

    # render environment
    def render(self, mode='human'):
        print('Step:', self.current_step, 'Net Worth:', self.net_worth)


# Make pandas data frames from coin CSVs
def get_data(list_of_coins):
    dfs = []

    min_len = len(pd.read_csv(list_of_coins[0])) - 2

    for i in range(len(list_of_coins)):
        if i == 0:
            continue

        curr_len = len(pd.read_csv(list_of_coins[i])) - 2

        if curr_len < min_len:
            min_len = curr_len

    for s in list_of_coins:
        df = pd.read_csv(s, skiprows=1)
        df = df.loc[0:min_len]
        df.sort_values(by='unix', ascending=True, inplace=True)
        df = df.reset_index()
        dfs.append(df)

    return dfs


# Coin CSV paths
listOfCoins = [
    './data/Binance_BTCUSDT_d.csv',
    './data/Binance_ETHUSDT_d.csv']

env = CryptoEnv(get_data(listOfCoins))

check_env(env, warn=True)

itr = int(input('number of iterations for A2C?\n'))

model = A2C("MlpPolicy", env, verbose=1).learn(itr)

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

itr = int(input('number of iterations for DQN?\n'))

model = DQN("MlpPolicy", env, verbose=1).learn(itr)

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

itr = int(input('number of iterations for PPO?\n'))

model = PPO("MlpPolicy", env, verbose=1).learn(itr)

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
