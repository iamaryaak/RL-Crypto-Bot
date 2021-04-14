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
ax.set_title('The Price of Bitcoin, 2017-2021')
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

# Add discrete fourier transforms
close_fft = np.fft.fft(np.asarray(df['close'].to_numpy()))
df['fft'] = close_fft
fft_list = np.asarray(df['fft'].tolist())
# Create a 3-component FFT #
fft_list_m10 = np.copy(fft_list)
fft_list_m10[3:-3] = 0
df['ifft_3'] = np.real(np.real(np.fft.ifft(fft_list_m10)))
# Done #
df = df.drop(['fft'], axis=1)
ax.plot('date', 'ifft_3', data=df, label='3-component FFT')

# Add a legend.
ax.legend()

# Plot!
plt.show()
