import gym
from gym import spaces
from gym.utils import seeding

import pandas as pd
import numpy as np
import random
from collections import deque

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

class CryptoEnv(gym.Env):

    metadata = {'render.modes': ['console']}

    # A custom Bitcoin trading environment
    def __init__(self, df, test=False, initial_balance=1000):
        # Define action space and state size and other custom parameters
        self.dfs = dfs
        self.test = test
        self.df_total_steps = 0
        if self.test:
            self.df_total_steps = 1314
        else:
            self.df_total_steps = 1000
        self.initial_balance = initial_balance

        # Action space 0 hold, 1 buy BTC, 2 buy ETC, 3 Sell
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)

        # Start with balance fully liquidated
        self.current_coin = 3

        # Define observation space
        self.observation_space = spaces.Box(0, 3, (10,), dtype=np.float32)

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.current_coin = 3
        if self.test:
            self.start_step = 1000
        else:
            self.start_step = 0

        self.end_step = self.df_total_steps

        self.current_step = self.start_step
        
        market_history_c1 = [self.dfs[0].loc[self.current_step, 'open']/100000000.0,
                            self.dfs[0].loc[self.current_step, 'high']/100000000.0,
                            self.dfs[0].loc[self.current_step, 'low']/100000000.0,
                            self.dfs[0].loc[self.current_step, 'close']/100000000.0,
                            self.dfs[0].loc[self.current_step, 'Volume USDT']/100000000.0,
                            self.dfs[1].loc[self.current_step, 'open']/100000000.0,
                            self.dfs[1].loc[self.current_step, 'high']/100000000.0,
                            self.dfs[1].loc[self.current_step, 'low']/100000000.0,
                            self.dfs[1].loc[self.current_step, 'close']/100000000.0,
                            self.dfs[1].loc[self.current_step, 'Volume USDT']/100000000.0]

        return np.array(market_history_c1).astype(np.float32)

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0

        # Set the current price to a random price between open and close
        priceOfBTC = random.uniform(self.dfs[0].loc[self.current_step, 'open'], self.dfs[0].loc[self.current_step, 'close'])
        priceOfETC = random.uniform(self.dfs[1].loc[self.current_step, 'open'], self.dfs[1].loc[self.current_step, 'close'])

        held_coin_price = 0

        if self.current_coin == 0:
            held_coin_price = priceOfBTC

        if self.current_coin == 1:
            held_coin_price = priceOfETC

        if action == 0: # Hold
            pass

        elif action == 1 and self.balance > 0:
            # Sell all crpto and buy BCT with 100% of current balance
            self.current_coin = 0
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * held_coin_price
            self.crypto_held -= self.crypto_sold

            self.current_coin = 0
            self.crypto_bought = self.balance / priceOfBTC
            self.balance -= self.crypto_bought * priceOfBTC
            self.crypto_held += self.crypto_bought

        elif action == 2 and self.balance > 0:
            # Sell all crpto and buy ETC with 100% of current balance
            self.current_coin = 0
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * held_coin_price
            self.crypto_held -= self.crypto_sold

            self.current_coin = 1
            self.crypto_bought = self.balance / priceOfETC
            self.balance -= self.crypto_bought * priceOfETC
            self.crypto_held += self.crypto_bought

        elif action == 3 and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.current_coin = 0
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * held_coin_price
            self.crypto_held -= self.crypto_sold

        if self.current_coin == 0:
            held_coin_price = priceOfBTC

        if self.current_coin == 1:
            held_coin_price = priceOfETC

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * held_coin_price

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        market_history_c1 = [self.dfs[0].loc[self.current_step, 'open']/100000000.0,
                            self.dfs[0].loc[self.current_step, 'high']/100000000.0,
                            self.dfs[0].loc[self.current_step, 'low']/100000000.0,
                            self.dfs[0].loc[self.current_step, 'close']/100000000.0,
                            self.dfs[0].loc[self.current_step, 'Volume USDT']/100000000.0,
                            self.dfs[1].loc[self.current_step, 'open']/100000000.0,
                            self.dfs[1].loc[self.current_step, 'high']/100000000.0,
                            self.dfs[1].loc[self.current_step, 'low']/100000000.0,
                            self.dfs[1].loc[self.current_step, 'close']/100000000.0,
                            self.dfs[1].loc[self.current_step, 'Volume USDT']/100000000.0]

        obs = np.array(market_history_c1).astype(np.float32)
        self.current_step += 1

        return obs, reward, self.current_step == self.df_total_steps, {}

    # render environment
    def render(self):
        print('Step:', self.current_step, 'Net Worth:', self.net_worth)

def GetData(listOfCoins):
    dfs = []

    minLen = len(pd.read_csv(listOfCoins[0])) - 2

    for i in range(len(listOfCoins)):
        if i == 0:
            continue

        currLen = len(pd.read_csv(listOfCoins[i])) - 2

        if currLen < minLen:
            minLen = currLen

    for s in listOfCoins:
        dfLen = len(pd.read_csv(s))

        df = pd.read_csv(s, skiprows=1)
        df = df.loc[0:minLen]
        df.sort_values(by = 'unix', ascending=True, inplace=True)
        df = df.reset_index()
        dfs.append(df)

    return dfs

listOfCoins = [
        './data/Binance_BTCUSDT_d.csv',
        './data/Binance_ETHUSDT_d.csv']

dfs = GetData(listOfCoins)

env = CryptoEnv(dfs)
testEnv = CryptoEnv(dfs, True)

itr = int(input('number of iterations for A2C?\n'))
model = A2C("MlpPolicy", env, verbose=1).learn(itr)

obs = testEnv.reset()
env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = testEnv.step(action)
    testEnv.render()

itr = int(input('number of iterations for DQN?\n'))
model = DQN("MlpPolicy", env, verbose=1).learn(itr)

obs = testEnv.reset()
env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = testEnv.step(action)
    testEnv.render()

itr = int(input('number of iterations for PPO?\n'))
model = PPO("MlpPolicy", env, verbose=1).learn(itr)

obs = testEnv.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = testEnv.step(action)
    testEnv.render()
