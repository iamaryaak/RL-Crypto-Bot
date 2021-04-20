import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts


class FatFluorescentOrangeSeahorse(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2016, 1, 1)  # Set Start Date
        self.SetCash(10000000)  # Set Strategy Cash
        self.jpm = self.AddEquity("JPM", Resolution.Hour)
        self.gs = self.AddEquity("GS", Resolution.Hour)
        self.ms = self.AddEquity("MS", Resolution.Hour)
        self.cs = self.AddEquity("CS", Resolution.Hour)
        self.check_jpm_gs = self.coint_test("JPM", "GS")
        self.check_jpm_ms = self.coint_test("JPM", "MS")
        self.check_jpm_cs = self.coint_test("JPM", "CS")
        self.check_ms_gs = self.coint_test("MS", "GS")
        self.check_cs_gs = self.coint_test("CS", "GS")
        self.check_ms_cs = self.coint_test("MS", "CS")

    def OnData(self, data):
        jpm_check = False
        gs_check = False
        cs_check = False
        ms_check = False
        if self.is_stationary("JPM", "GS") < 0.05:
            if (not jpm_check and not gs_check) or (
                    jpm_check and gs_check) and not ms_check and self.check_jpm_gs < 0.05:
                mu, sigma = self.zstats("JPM", "GS")
                price1 = self.History(self.Symbol("JPM"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                price2 = self.History(self.Symbol("GS"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                if ((price1 - price2) - mu) / sigma > 1.5:
                    self.setHoldings("JPM", -0.1)
                    self.SetHolding("GS", 0.1)
                else:
                    self.setHoldings("JPM", 0)
                    self.SetHolding("GS", 0)
        if self.is_stationary("JPM", "MS") < 0.05:
            if (not jpm_check and not ms_check) or (
                    (jpm_check and ms_check) and not gs_check) and self.check_jpm_ms < 0.05:
                mu, sigma = self.zstats("JPM", "MS")
                price1 = self.History(self.Symbol("JPM"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                price2 = self.History(self.Symbol("MS"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                if ((price1 - price2) - mu) / sigma > 1.5:
                    self.setHoldings("JPM", -0.1)
                    self.SetHolding("MS", 0.1)
                else:
                    self.setHoldings("JPM", 0)
                    self.SetHolding("MS", 0)
        if self.is_stationary("JPM", "CS") < 0.05:
            if (not jpm_check and not cs_check) or (
                    (jpm_check and cs_check) and not ms_check) and self.check_jpm_cs < 0.05:
                mu, sigma = self.zstats("JPM", "CS")
                price1 = self.History(self.Symbol("JPM"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                price2 = self.History(self.Symbol("CS"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                if ((price1 - price2) - mu) / sigma > 1.5:
                    self.setHoldings("JPM", -0.1)
                    self.SetHolding("CS", 0.1)
                else:
                    self.setHoldings("JPM", 0)
                    self.SetHolding("CS", 0)
        if self.is_stationary("GS", "MS") < 0.05:
            if (not gs_check and not ms_check) or (
                    (gs_check and ms_check) and not jpm_check) and self.check_ms_gs < 0.05:
                mu, sigma = self.zstats("GS", "MS")
                price1 = self.History(self.Symbol("GS"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                price2 = self.History(self.Symbol("MS"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                if ((price1 - price2) - mu) / sigma > 1.5:
                    self.setHoldings("GS", -0.1)
                    self.SetHolding("MS", 0.1)
                else:
                    self.setHoldings("GS", 0)
                    self.SetHolding("MS", 0)
        if self.is_stationary("GS", "CS") < 0.05:
            if (not gs_check and not cs_check) or (
                    (gs_check and cs_check) and not ms_check) and self.check_cs_gs < 0.05:
                mu, sigma = self.zstats("GS", "CS")
                price1 = self.History(self.Symbol("GS"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                price2 = self.History(self.Symbol("CS"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                if ((price1 - price2) - mu) / sigma > 1.5:
                    self.setHoldings("GS", -0.1)
                    self.SetHolding("CS", 0.1)
                else:
                    self.setHoldings("GS", 0)
                    self.SetHolding("CS", 0)
        if self.is_stationary("CS", "MS") < 0.05:
            if (not cs_check and not ms_check) or (
                    (cs_check and ms_check) and not jpm_check) and self.check_ms_cs < 0.05:
                mu, sigma = self.zstats("CS", "MS")
                price1 = self.History(self.Symbol("CS"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                price2 = self.History(self.Symbol("MS"), timedelta(2), Resolution.Hour)['close'].pct_change().loc[0][
                    'close']
                if ((price1 - price2) - mu) / sigma > 1.5:
                    self.setHoldings("CS", -0.1)
                    self.SetHolding("MS", 0.1)
                else:
                    self.setHoldings("CS", 0)
                    self.SetHolding("MS", 0)

    def coint_test(self, sec1, sec2):
        df1 = self.History(self.Symbol(sec1), timedelta(150), Resolution.Hour)['close']
        df2 = self.History(self.Symbol(sec2), timedelta(150), Resolution.Hour)['close']
        return ts.coint(df1, df2)[1]

    def is_stationary(self, sec1, sec2):
        df1 = self.History(self.Symbol(sec1), timedelta(150), Resolution.Hour)['close'].pct_change()
        df2 = self.History(self.Symbol(sec2), timedelta(150), Resolution.Hour)['close'].pct_change()
        dfdiff = df1 - df2
        return ts.adfuller(dfdiff)[1]

    def zstats(self, sec1, sec2):
        df1 = self.History(self.Symbol(sec1), timedelta(150), Resolution.Hour)['close'].pct_change()
        df2 = self.History(self.Symbol(sec2), timedelta(150), Resolution.Hour)['close'].pct_change()
        dfdiff = df1 - df2
        mu = self.dfdiff.mean()
        sigma = self.dfdiff.std()
        return mu, sigma