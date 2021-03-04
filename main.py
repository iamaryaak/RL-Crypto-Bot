
import pandas as pd
import tensortrade.env.default as default

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.instruments import USD, BTC, ETH, LTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order

cdd = CryptoDataDownload()

bitfinex_data = pd.concat([
    cdd.fetch("Bitfinex", "USD", "BTC", "1h").add_prefix("BTC:"),
    cdd.fetch("Bitfinex", "USD", "ETH", "1h").add_prefix("ETH:")
], axis=1)

bitstamp_data = pd.concat([
    cdd.fetch("Bitstamp", "USD", "BTC", "1h").add_prefix("BTC:"),
    cdd.fetch("Bitstamp", "USD", "LTC", "1h").add_prefix("LTC:")
], axis=1)

print(bitfinex_data.head())

