import pandas as pd
import numpy as np
import random
from collections import deque

class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50):
        # Define action space and state size and other custom parameters
        self.dfs = dfs
        self.df_total_steps = len(self.dfs[0])-1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        # Action space 0 hold, 1 buy BTC, 2 buy ETC, 3 Sell
        self.action_space = np.array([0, 1, 2, 3])

        # Orders history contains the balance, net_worth, current Crypto held, BTC bought, ETC bought, BTC sold, ETC sold, ETC held, BTC held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)
        
        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 19)

        # Start with balance fully liquidated
        self.current_coin = 3;

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        if env_steps_size > 0: # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            
        self.current_step = self.start_step
        
        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.current_coin, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.dfs[0].loc[current_step, 'open'],
                                        self.dfs[0].loc[current_step, 'high'],
                                        self.dfs[0].loc[current_step, 'low'],
                                        self.dfs[0].loc[current_step, 'close'],
                                        self.dfs[0].loc[current_step, 'Volume USDT'],
                                        self.dfs[1].loc[current_step, 'open'],
                                        self.dfs[1].loc[current_step, 'high'],
                                        self.dfs[1].loc[current_step, 'low'],
                                        self.dfs[1].loc[current_step, 'close'],
                                        self.dfs[1].loc[current_step, 'Volume USDT']])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.dfs[0].loc[self.current_step, 'open'],
                                    self.dfs[0].loc[self.current_step, 'high'],
                                    self.dfs[0].loc[self.current_step, 'low'],
                                    self.dfs[0].loc[self.current_step, 'close'],
                                    self.dfs[0].loc[self.current_step, 'Volume USDT'],
                                    self.dfs[1].loc[self.current_step, 'open'],
                                    self.dfs[1].loc[self.current_step, 'high'],
                                    self.dfs[1].loc[self.current_step, 'low'],
                                    self.dfs[1].loc[self.current_step, 'close'],
                                    self.dfs[1].loc[self.current_step, 'Volume USDT']])

        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

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

            self.current_coin = 0;
            self.crypto_bought = self.balance / priceOfBTC
            self.balance -= self.crypto_bought * priceOfBTC
            self.crypto_held += self.crypto_bought

        elif action == 2 and self.balance > 0:
            # Sell all crpto and buy ETC with 100% of current balance
            self.current_coin = 0
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * held_coin_price
            self.crypto_held -= self.crypto_sold

            self.current_coin = 1;
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

        self.orders_history.append([self.balance, self.net_worth, self.current_coin, self.crypto_bought, self.crypto_sold, self.crypto_held])

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        obs = self._next_observation()
        
        return obs, reward, False

    # render environment
    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')

 
def Random_games(env, train_episodes = 50, training_batch_size=500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        while True:
            env.render()

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)

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
        dfs.append(df)

    return dfs;

listOfCoins = [
        '../data/Binance_BTCUSDT_d.csv',
        '../data/Binance_ETHUSDT_d.csv',]

dfs = GetData(listOfCoins)

lookback_window_size = 10
train_df = dfs
test_df = dfs

train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size)

Random_games(train_env, train_episodes = 10, training_batch_size=500)
