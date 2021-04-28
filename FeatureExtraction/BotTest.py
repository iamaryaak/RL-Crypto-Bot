# BASELINE BOT #
# This bot takes daily price data for one or more cryptocurrencies and trades them based on MACD indicators.
# TODO: Write a short set of instructions.
# Dependencies:
#   pandas - reading/manipulating data
#   matplotlib - graphing things
#   numpy - working with times
#   glob, os - importing files

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import glob
import os

# Read all CSVs in the current working directory into a new pandas dataframe
# Note: I had to take out "https://www.CryptoDataDownload.com" from the first line of each CSV in order for this to work
# Array of date and close price dataframes to concatenate
dataframes = []
# Make dates dataframe from BTC data
# Note: We assume that every CSV covers the same date range. I believe this is true for everything pulled from CDD.
dates = pd.read_csv('Binance_BTC_USDT_OHLCV.csv')
# Drop superfluous columns
for column in dates.columns:
    if column != 'date':
        dates = dates.drop(column, axis=1)
dataframes.append(dates)
# Make close price dataframes from every CSV in the current working directory
for file in glob.glob(os.path.join(os.getcwd(), "*.csv")):
    df = pd.read_csv(file)
    # Drop superfluous columns
    for column in df.columns:
        if column != 'close':
            df = df.drop(column, axis=1)
    # Append first three letters of coin name to the close column to make it unique
    # Note: We assume the 18th to last letter of the file name is the start of the coin name.
    # TODO: Use regex to read the coin name until an underscore
    df.rename(columns={'close': 'close_' + file[-18:-15].lower()}, inplace=True)
    dataframes.append(df)
# Concatenate!
df = pd.concat(dataframes, axis=1)
# Reverse the dataframe while keeping the index intact
df = df.loc[::-1].set_index(df.index)
# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Create a figure and a set of subplots
fig, ax = plt.subplots()
# Plot date vs. close
for column in df.columns:
    if column != 'date':
        ax.plot('date', column, data=df, label=str(column)[-3:].upper())
# Set titles
ax.set_title('Trading on MACD Indicators, 2017-2021')
ax.set_xlabel('Date')
ax.set_ylabel('USDT')

# Draw major ticks every six months
fmt_half_year = mdates.MonthLocator(interval=6)
ax.xaxis.set_major_locator(fmt_half_year)
# Draw minor ticks every month
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)

# Set month labels
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Set x-axis view limits
datemin = np.datetime64(df['date'].iloc[0], 'M')
datemax = np.datetime64(df['date'].iloc[-1], 'M') + np.timedelta64(1, 'M')
ax.set_xlim(datemin, datemax)

# Format the hover box
ax.format_xdata = mdates.DateFormatter('%Y-%m')
ax.format_ydata = lambda x: f'${x:.2f}'

# Enable grid
ax.grid(True)

# Rotate the month labels
fig.autofmt_xdate()


# Exponential Moving Average
def ema(period, col):
    df['ema' + str(period)] = df[column].ewm(ignore_na=False, min_periods=period, com=period, adjust=True).mean()


# Moving Average Convergence Divergence
def macd(dataframe, coin):
    # Define long (26), short (12), and signal (9) periods
    period_long = 26
    period_short = 12
    period_signal = 9
    # Make array of EMA columns to drop later
    remove_cols = []
    # Compute short and long EMA columns
    if not 'ema' + str(period_long) in dataframe.columns:
        ema(period_long, 'close_' + coin)
        remove_cols.append('ema' + str(period_long))
    if not 'ema' + str(period_short) in dataframe.columns:
        ema(period_short, 'close_' + coin)
        remove_cols.append('ema' + str(period_short))
    # Compute MACD value and signal line columns
    dataframe['macd_val_' + coin] = dataframe['ema' + str(period_short)] - dataframe[
        'ema' + str(period_long)]
    dataframe['macd_signal_line_' + coin] = \
        dataframe['macd_val_' + coin].ewm(
            ignore_na=False, min_periods=0, com=period_signal, adjust=True).mean()
    # Make (MACD - Signal) column
    df['macd_v_signal_' + coin] = df['macd_val_' + coin] - df['macd_signal_line_' + coin]
    # Drop columns
    dataframe = dataframe.drop(remove_cols, axis=1)
    return dataframe


for column in df.columns:
    # Get coin name from column
    coin_name = str(column)[-3:]
    if column != 'date':
        # Make (MACD - Signal) column
        df = macd(df, coin_name)
        # Plot MACD value and signal line
        ax.plot('date', 'macd_val_' + coin_name, data=df, label=coin_name.upper() + ' MACD')
        ax.plot('date', 'macd_signal_line_' + coin_name, data=df, label=coin_name.upper() + ' Signal')
        # Drop value and signal columns
        df = df.drop(['macd_val_' + coin_name, 'macd_signal_line_' + coin_name], axis=1)

print(df)
print()


# Create Portfolio class
class Portfolio:
    USD = 10000  # Starting USD
    BTC = 0  # Starting BTC
    BTC_val = 0  # USD value of BTC
    ETH = 0  # Starting ETH
    ETH_val = 0  # USD value of ETH


# Create portfolio object
portfolio = Portfolio()


# Get portfolio value
def get_value():
    return portfolio.USD + portfolio.BTC_val + portfolio.ETH_val

print(df)
# Instantiate portfolio holdings column
df['portfolio'] = ''
df.at[0, 'portfolio'] = get_value()
print('Portfolio: $' + '%.3f' % get_value())

# Decide how much risk we want to take on
# TODO: Experiment with these values to find what gets us the best % return
#           THIS IS THE HIGHEST PRIORITY TASK ^^ none of the others ones are strictly necessary :)
max_purchase = 5000  # The maximum BTC we're willing to purchase when things are trending up
max_sale = 2500  # The maximum BTC we're willing to sell when things are trending down
# Populate portfolio column, from row 1 to len(df) - 1
for row in range(1, len(df)):
    # Booleans to capture our actions
    made_purchase = False
    made_sale = False

    # TODO: Try combining these two parts
    # Part 1: Trade BTC
    purchase = 0
    sale = 0
    # Buy BTC when MACD crosses above its signal line
    if df.at[row - 1, 'macd_v_signal_btc'] <= 0 and df.at[row, 'macd_v_signal_btc'] > 0:
        purchase = min(max_purchase, portfolio.USD)
        portfolio.USD -= purchase
        portfolio.BTC += (purchase / df.at[row, 'close_btc'])
    # Sell BTC when MACD crosses below its signal line
    if df.at[row - 1, 'macd_v_signal_btc'] >= 0 and df.at[row, 'macd_v_signal_btc'] < 0:
        sale = min(max_sale, portfolio.BTC_val)
        portfolio.USD += sale
        portfolio.BTC -= (sale / df.at[row, 'close_btc'])
    # Update BTC holdings
    portfolio.BTC_val = portfolio.BTC * df.at[row, 'close_btc']
    # Capture action(s)
    if purchase > 0:
        made_purchase = True
        print('Buy $' + str(purchase), 'BTC on', str(df.at[row, 'date'])[0:10])
    if sale > 0:
        made_sale = True
        print('Sell $' + str(sale), 'BTC on', str(df.at[row, 'date'])[0:10])

    # Part 2: Trade ETH
    purchase = 0
    sale = 0
    # Buy ETH when MACD crosses above its signal line
    if df.at[row - 1, 'macd_v_signal_eth'] <= 0 and df.at[row, 'macd_v_signal_eth'] > 0:
        purchase = min(max_purchase, portfolio.USD)
        portfolio.USD -= purchase
        portfolio.ETH += (purchase / df.at[row, 'close_eth'])
    # Sell ETH when MACD crosses below its signal line
    if df.at[row - 1, 'macd_v_signal_eth'] >= 0 and df.at[row, 'macd_v_signal_eth'] < 0:
        sale = min(max_sale, portfolio.ETH_val)
        portfolio.USD += sale
        portfolio.ETH -= (sale / df.at[row, 'close_eth'])
    # Update ETH holdings
    portfolio.ETH_val = portfolio.ETH * df.at[row, 'close_eth']
    # Capture action(s)
    if purchase > 0:
        made_purchase = True
        print('Buy $' + str(purchase), 'ETH on', str(df.at[row, 'date'])[0:10])
    if sale > 0:
        made_sale = True
        print('Sell $' + str(sale), 'ETH on', str(df.at[row, 'date'])[0:10])

    df.at[row, 'portfolio'] = get_value()
    if made_purchase:
        ax.plot(df.at[row, 'date'], df.at[row, 'portfolio'], 'go', markersize=5)
    if made_sale:
        ax.plot(df.at[row, 'date'], df.at[row, 'portfolio'], 'ro', markersize=5)
    # print('Portfolio: $' + '%.3f' % get_value())

# Print final portfolio value
print('Portfolio: $' + '%.3f' % get_value())
print('Performance:', '%.3f' % ((get_value() / 100) - 100) + "% (Compare to BTC's appreciation of 1,232.32%)")

# Plot portfolio value
ax.plot('date', 'portfolio', data=df, label='Portfolio Value')

# Add a legend.
ax.legend()

# Plot!
plt.show()