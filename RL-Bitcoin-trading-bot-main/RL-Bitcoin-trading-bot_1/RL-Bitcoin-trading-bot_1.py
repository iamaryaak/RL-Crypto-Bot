import pandas as pd
import numpy as np
import random
from collections import deque

class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, dfs, initial_balance=1000, lookback_window_size=50):
        # Define action space and state size and other custom parameters
        self.dfs = dfs;
        self.df_total_steps = len(self.dfs[0])-1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 10)

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        if env_steps_size > 0: # used for training dataset
            print(self.lookback_window_size, self.df_total_steps, env_steps_size)
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'open'],
                                        self.df.loc[current_step, 'high'],
                                        self.df.loc[current_step, 'low'],
                                        self.df.loc[current_step, 'close'],
                                        self.df.loc[current_step, 'Volume USDT']
                                        ])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'open'],
                                    self.df.loc[self.current_step, 'high'],
                                    self.df.loc[self.current_step, 'low'],
                                    self.df.loc[self.current_step, 'close'],
                                    self.df.loc[self.current_step, 'Volume USDT']
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'open'],
            self.df.loc[self.current_step, 'close'])
        
        if action == 0: # Hold
            pass
        
        elif action == 1 and self.balance > 0:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought

        elif action == 2 and self.crypto_held>0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False

        obs = self._next_observation()
        
        return obs, reward, done

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
        '../data/Binance_ADAUSDT_d.csv',
        '../data/Binance_EOSUSDT_d.csv',
        '../data/Binance_NEOUSDT_d.csv',
        '../data/Binance_XRPUSDT_d.csv',
        '../data/Binance_BNBUSDT_d.csv',
        '../data/Binance_ETCUSDT_d.csv',
        '../data/Binance_QTUMUSDT_d.csv',
        '../data/Binance_ZECUSDT_d.csv',
        '../data/Binance_ETHUSDT_d.csv',
        '../data/Binance_TRXUSDT_d.csv',
        '../data/Binance_BTTUSDT_d.csv',
        '../data/Binance_LINKUSDT_d.csv',
        '../data/Binance_DASHUSDT_d.csv',
        '../data/Binance_XLMUSDT_d.csv',
        '../data/Binance_LTCUSDT_d.csv',
        '../data/Binance_XMRUSDT_d.csv' ]

dfs = GetData(listOfCoins)

lookback_window_size = 10
train_dfs = dfs
test_dfs = dfs # 30 days

train_env = CustomEnv(train_dfs, lookback_window_size=lookback_window_size)
test_env = CustomEnv(test_dfs, lookback_window_size=lookback_window_size)

'''Random_games(train_env, train_episodes = 10, training_batch_size=500)'''
