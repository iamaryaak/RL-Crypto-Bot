import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Import Historic Stock charts - Upload a CSV if not already done
# from google.colab import files
# files.upload()

AAPL = pd.read_csv('Binance_BTCUSDT_d.csv')
# Set the date as the index
AAPL = AAPL.set_index(pd.DatetimeIndex(AAPL['Date'].values))

# Show the data
AAPL

# Truncate the data until 26th Dec 2006
AAPL = AAPL[0:60] # 60 rows mark until 26th Dec 2006
max_close = AAPL['Close'].max()
min_close = AAPL['Close'].min()

print(AAPL)
print("Max Close Price = ", max_close)
print("Min Close Price = ", min_close)

# Calculate the three moving averages

# Calculate the short term exponential moving average (SEMA)
shortPeriod = 5
Short_EMA = AAPL.Close.ewm(span=shortPeriod, adjust= False).mean()

# Calculate the medium exponential moving average (MEMA)
medPeriod = 21
Med_EMA = AAPL.Close.ewm(span=medPeriod, adjust= False).mean()

# Calculate the long term exponential moving average (SEMA)
longPeriod = 63
Long_EMA = AAPL.Close.ewm(span=longPeriod, adjust= False).mean()

# Store EMAs in the dataset
AAPL['SEMA'] = Short_EMA
AAPL['MEMA'] = Med_EMA
AAPL['LEMA'] = Long_EMA
AAPL


# Investment signal
def investment_signal(data):
    sigPriceToBuy = []
    sigPriceToSell = []
    flagLong = 0  # flag to check when STMA crosses over LTMA
    flagShort = 0
    for i in range(len(data)):
        if data['MEMA'][i] < data['LEMA'][i] and data['SEMA'][i] < data['MEMA'][i] and flagLong == 0 and flagShort == 0:
            sigPriceToBuy.append(data['Close'][i])  # good time to buy
            sigPriceToSell.append(np.nan)
            flagShort = 1

        elif data['SEMA'][i] > data['MEMA'][i] and flagShort == 1:
            sigPriceToBuy.append(np.nan)
            sigPriceToSell.append(data['Close'][i])  # good time to sell
            flagShort = 0

        elif data['MEMA'][i] > data['LEMA'][i] and data['SEMA'][i] > data['MEMA'][
            i] and flagLong == 0 and flagShort == 0:
            sigPriceToBuy.append(np.nan)
            sigPriceToSell.append(np.nan)
            flagLong = 1

        elif data['SEMA'][i] < data['MEMA'][i] and flagLong == 1:
            sigPriceToBuy.append(np.nan)
            sigPriceToSell.append(data['Close'][i])  # good time to sell
            flagLong = 0

        else:
            sigPriceToBuy.append(np.nan)
            sigPriceToSell.append(np.nan)

    return (sigPriceToBuy, sigPriceToSell)

# Store the Buy and Sell data
buy_sell = investment_signal(AAPL)
AAPL['Buy_Signal_Price'] = buy_sell[0]
AAPL['Sell_Signal_Price'] = buy_sell[1]
AAPL