import yfinance as yf
import pandas as pd
import regex as re
import math
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import datetime as dt
import json
import requests
import psycopg2 as sql
import sched, time
from sqlalchemy import create_engine
# Server UN: lldwretqol
# Server Pass: 55H20SUU8026707S$

def BSM(S, K, r, T, vol, opType):
    t = T/365.25
    d1 = (math.log(S / K) + (r + vol ** 2 / 2) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)

    if(opType == "C"):
        return S *  norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    else:
        return (K * math.exp(-r * t) * norm.cdf(-d2) - S *  norm.cdf(-d1))
    
def IV(S, K, r, T, v, opType, opPrc, max_iter, tolerance):
    if(opPrc == 0):
        return np.nan
    OpVal = np.inf
    vol = v
    dv = 0.0001
    tolerance = 0.0001
    for i in range(max_iter):
        OpVal = BSM(S, K, r, T, vol, opType)
        OV1 = BSM(S, K, r, T, vol - dv, opType)
        OV2 = BSM(S, K, r, T, vol + dv, opType)
        dVol = (OV1 - OV2)/(2* dv)
        if(dVol == 0):
            return 0
        elif(abs(opPrc - OpVal) <= tolerance):
            return vol
        else:
            ddv = (opPrc - OpVal)/dVol
            vol -= ddv

    return vol

def D1(S, K, r, T, v):
    if(v == 0):
        return np.inf
    elif(v == np.nan):
        return np.nan

    return (math.log(S / K) + (r + v ** 2 / 2) * T) / (v * math.sqrt(T))


def CalcDelta(d1, opType):
    if(d1 == np.inf):
        if(opType == "C"):
            return 1
        else: return -1
    elif(d1 == np.nan):
        return 0
    delta = norm.cdf(d1)
    if(opType == "C"):
        return delta
    else: return delta - 1

def CalcGamma(S, T, v, d1):
        if(v == 0 or v == np.nan):
            return 0
        g = math.exp(- d1 ** 2 /2)/(S * v * math.sqrt(2 * np.pi * T))
        return g

def CalcSpeed(S, T, v, G, d1):
    if(v ==0 or v == np.nan):
        return 0
    return - G/S * (d1/(v * math.sqrt(T) + 1))


def ScrapeOptions(spy: yf.Ticker, vix: yf.Ticker, SExpDates: list(), VExpDates: list(), RFR: float, SpyHist: list(), LastTrades: list()):
    Options = list()
    TimeFound = None
    Spr = None
    Vpr = None
    Spr = spy.fast_info.last_price
    Vpr = vix.fast_info.last_price
    while True:
        for i in range(len(SExpDates)):
            SOpt = spy.option_chain(SExpDates[i])
            VOpt = vix.option_chain(VExpDates[i])
            Options.extend([SOpt.calls, SOpt.puts, VOpt.calls, VOpt.puts])
        TimeFound = datetime.today() - pd.Timedelta(minutes = -15)

        AllOptions = pd.concat(Options, axis = 0)
        Trades = list()
        Trades.extend(AllOptions["lastTradeDate"])
        if(len(LastTrades) == 0 or Trades != LastTrades):
            break
        else: 
            print(f"Trade values are same as last at {datetime.utcnow()}")
            time.sleep(0.1)

    AllOptions["Symbol"] = AllOptions["contractSymbol"].str.extract('^(\D+)')
    AllOptions["Underlying"] = AllOptions.apply(lambda row: SpyHist[-15] if row["Symbol"] == "SPY" else Vpr, axis = 1)
    AllOptions["QuoteTime"] = TimeFound
    AllOptions["ExpiryDate"] = AllOptions.apply(lambda row: row["contractSymbol"][len(row["Symbol"]):len(row["Symbol"]) + 6], axis = 1)
    AllOptions["ExpiryDate"] = pd.to_datetime(AllOptions["ExpiryDate"],format= "%y%m%d") + pd.Timedelta(hours= 21)
    AllOptions["ExpiryDate"] = AllOptions["ExpiryDate"].dt.to_pydatetime()
    AllOptions["OpType"] = AllOptions.apply(lambda row: row["contractSymbol"][len(row["Symbol"]) + 6], axis = 1)

    AllOptions["DTE"] = AllOptions.apply(lambda row: (row["ExpiryDate"] - TimeFound).total_seconds()/(27000.0) 
                                        if (row["ExpiryDate"] - TimeFound).total_seconds()/(27000.0) < 1 else 
                                        (row["ExpiryDate"] - TimeFound).total_seconds()/(27000.0), axis = 1)
    """ AllOptions["IV"] = AllOptions.apply(lambda row: 
                                        IV(row["Underlying"], row["strike"], RFR, 
                                        row["DTE"]/365.25, 
                                        row["impliedVolatility"], row["OpType"], (row["bid"] + row["ask"])/2, 100, 0.0001), axis = 1) """
    AllOptions["IV"] = AllOptions["impliedVolatility"]
    AllOptions["RiskFreeRate"] = RFR
    AllOptions["D1"] = AllOptions.apply(lambda row: D1(row["Underlying"], row["strike"], RFR, 
                                        row["DTE"]/365.25, 
                                        row["IV"]), axis = 1)
    AllOptions["Delta"] = AllOptions.apply(lambda row: CalcDelta(row["D1"], row["OpType"]), axis = 1)
    AllOptions["Gamma"] = AllOptions.apply(
        lambda row: CalcGamma(row["Underlying"],
                            row["DTE"]/365.25, 
                            row["IV"], row["D1"]), axis = 1)
    AllOptions["Speed"] = AllOptions.apply(
        lambda row: CalcSpeed(row["Underlying"],
                            row["DTE"]/365.25, 
                            row["IV"],row["Gamma"], row["D1"]), axis = 1)

    AllOptions = AllOptions.drop(columns=["contractSymbol", "contractSize", "currency", "impliedVolatility"])

    engine = create_engine('postgresql+psycopg2://dzdgvnslqh:DYPKDAcE9n@p4BH@histdsv-server.postgres.database.azure.com/postgres')
    AllOptions.to_sql("HistoricalData", engine, if_exists="append")
    SpyHist.append(Spr)
    SpyHist.pop(0)
    LastTrades = list()
    LastTrades.extend(AllOptions["lastTradeDate"])
    return SpyHist, LastTrades

commands = """
        CREATE TABLE SPY (
            QuoteTime timestamp PRIMARY KEY,
            Underlying numeric NOT NULL,
            OpType varchar(1) NOT NULL,
            Strike numeric NOT NULL,
            Expiry timestamp NOT NULL,
            DTE numeric NOT NULL,
            Strike numeric NOT NULL,
            OpenInterest int,
            LastTrade timestamp NOT NULL,
            Bid numeric NOT NULL,
            Ask numeric NOT NULL,
            Volume numeric NOT NULL,
            RFR numeric NOT NULL,
            IV numeric NOT NULL,
            Delta numeric NOT NULL,
            Gamma numeric NOT NULL,
            Speed numeric NOT NULL
        )
        """

def main():
    spy = yf.Ticker("SPY")
    vix = yf.Ticker("^VIX")
    res = requests.get("https://markets.newyorkfed.org/api/rates/secured/all/latest.json")
    rates = json.loads(res.content)
    RFR = 0
    for i in range(len(rates["refRates"])):
        if(rates["refRates"][i]["type"] == "SOFR"):
            RFR = rates["refRates"][i]["percentRate"]/100.0 
    dtn = datetime.utcnow()
    future = datetime(dtn.year,dtn.month,dtn.day,14,45)

    if(dtn.hour < 14 or dtn.hour == 14 and dtn.minute < 45):
        print(f"Sleeping for {(future - dtn).total_seconds()/60.0} minutes")
        time.sleep((future - dtn).total_seconds())
    SExpDates = list()
    VExpDates = list()
    SExpDates.extend(spy.options[0:2])
    VExpDates.extend(vix.options[0:2])
    
    
    
    future = datetime(dtn.year,dtn.month,dtn.day,21,15,1)
    LastTrades = list()
    if(datetime.utcnow().second != 0):
        print("Waiting until next minute to begin")
        dt = datetime.utcnow()
        st = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0) + timedelta(minutes=1)
        time.sleep((st - datetime.utcnow()).total_seconds())

    Shist = spy.history("1d", "1m")
    Hist = list()
    Hist.extend(Shist["Open"].tail(15))
    while datetime.utcnow() <= future:
        starttime = time.time()
        Hist, LastTrades = ScrapeOptions(spy, vix, SExpDates, VExpDates, RFR, Hist, LastTrades)
        time.sleep(60.0 - ((time.time() - starttime) % 60.0))
        
    print("Finished scraping today's data")

if __name__ == "__main__":
    main()