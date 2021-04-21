import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Read BTC-USD CSV into a dataframe
# Note: I had to take out https://www.CryptoDataDownload.com from the first line in order for this to work
df = pd.read_csv('Binance_BTC_USDT_OHLCV.csv')
# Reverse the order of values
df.sort_values(by='unix', ascending=True, inplace=True)
# Convert unix time to datetime
df['date'] = pd.to_datetime(df['date'])

# Create a figure and a set of subplots
fig, ax = plt.subplots()
# Plot date vs. close
ax.plot('date', 'close', data=df, label='Daily price at close')
# Set titles
ax.set_title('Trading BTC on MACD Indicators, 2017-2021')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USDT)')

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
def ema(period=12, column='close'):
    df['ema' + str(period)] = df[column].ewm(ignore_na=False, min_periods=period, com=period, adjust=True).mean()


# Moving Average Convergence Divergence
# Define long (26), short (12), and signal (9) periods
period_long = 26
period_short = 12
period_signal = 9
# Make array of EMA columns to drop later
remove_cols = []
# Compute short and long EMA columns
if not 'ema' + str(period_long) in df.columns:
    ema(period_long)
    remove_cols.append('ema' + str(period_long))
if not 'ema' + str(period_short) in df.columns:
    ema(period_short)
    remove_cols.append('ema' + str(period_short))
# Compute MACD value and signal line columns
df['macd_val'] = df['ema' + str(period_short)] - df['ema' + str(period_long)]
df['macd_signal_line'] = df['macd_val'].ewm(ignore_na=False, min_periods=0, com=period_signal, adjust=True).mean()
# Drop EMA columns
df = df.drop(remove_cols, axis=1)

# Plot MACD value and signal line
ax.plot('date', 'macd_val', data=df, label='MACD Value')
ax.plot('date', 'macd_signal_line', data=df, label='MACD Signal')

# Make (MACD - Signal) column
df['macd_v_signal'] = df['macd_val'] - df['macd_signal_line']


# Create Portfolio class
class Portfolio:
    USD = 10000  # Starting USD
    BTC = 0  # Starting BTC
    BTC_val = 0  # USD value of BTC


# Create portfolio object
portfolio = Portfolio()

# Instantiate portfolio holdings column
df['portfolio'] = ''
df.at[len(df) - 1, 'portfolio'] = portfolio.USD + portfolio.BTC_val
print('Portfolio: $' + '%.3f' % (portfolio.USD + portfolio.BTC_val))
# Decide how much risk we want to take on
max_purchase = 1000  # The maximum BTC we're willing to purchase when things are trending up
max_sale = 1000  # The maximum BTC we're willing to sell when things are trending down
# Populate portfolio column, from row len(df) - 2 to 0
for offset in range(2, len(df) + 1):
    purchase = 0
    sale = 0
    # Buy BTC when MACD crosses above its signal line
    if df.at[(len(df) - offset) + 1, 'macd_v_signal'] <= 0 and df.at[(len(df) - offset), 'macd_v_signal'] > 0:
        purchase = min(max_purchase, portfolio.USD)
        portfolio.USD -= purchase
        portfolio.BTC += (purchase / df.at[len(df) - offset, 'close'])
        print('Buy $' + str(purchase), 'BTC on', str(df.at[len(df) - offset, 'date'])[0:10])
    # Sell BTC when MACD crosses below its signal line
    if df.at[(len(df) - offset) + 1, 'macd_v_signal'] >= 0 and df.at[(len(df) - offset), 'macd_v_signal'] < 0:
        sale = min(max_sale, portfolio.BTC_val)
        portfolio.USD += sale
        portfolio.BTC -= (sale / df.at[len(df) - offset, 'close'])
        print('Sell $' + str(sale), 'BTC on', str(df.at[len(df) - offset, 'date'])[0:10])
    # Update portfolio
    portfolio.BTC_val = portfolio.BTC * df.at[len(df) - offset, 'close']
    df.at[len(df) - offset, 'portfolio'] = portfolio.USD + portfolio.BTC_val
    if purchase > 0:
        ax.plot(df.at[len(df) - offset, 'date'], df.at[len(df) - offset, 'portfolio'], 'go', markersize=5)
    if sale > 0:
        ax.plot(df.at[len(df) - offset, 'date'], df.at[len(df) - offset, 'portfolio'], 'ro', markersize=5)
    # print('Portfolio: $' + '%.3f' % (portfolio.USD + portfolio.BTC_val))
# Print final portfolio value
print('Portfolio: $' + '%.3f' % (portfolio.USD + portfolio.BTC_val))

ax.plot('date', 'portfolio', data=df, label='Portfolio Value')

# Add a legend.
ax.legend()

# Plot!
plt.show()
