import pandas as pd
import numpy as np
import os
from baseModule.dayVol import EWMA, GARCH


raw_data = pd.read_csv("data/NSQ_1min.csv", parse_dates=["Date"])
close = pd.read_csv("data/NSQ_close.csv", parse_dates=["Local_Date_Time"])

def get_stock_data(stock_RIC, aggre_min=10, start_time=None, end_time=None):
    
    """
        input:
            stock_RIC: RIC code to retrieve cleaned data, (i.e. AAPL.O)
            aggre_min: minute interval length to aggregate/resample, default 10
            dayVol_method: specify the methods to fit and predict daily volatility
            start_time: specify the start date of the dataframe, inclusive, yyyy-mm-dd hh:MM:ss
            end_time: specify the end date of the dataframe, inclusive, yyyy-mm-dd hh:MM:ss
                
        output:
            full_data: preprocessed dataframe containing timeseries information for specified
                stock_RIC aggregated at given minute interval from start_date to end_date.
            dayVol: daily volatility calculated by summing volatility at aggregations during the day
    """
    target_data = raw_data.loc[raw_data.RIC == stock_RIC]
    
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    
    start_date = pd.to_datetime(start_time.date())
    end_date = pd.to_datetime(end_time.date())
    
    ## Add Auction Price information
    data = pd.merge(target_data, close,
                    left_on=["Date", "RIC"], right_on=["Local_Date_Time", "RIC"],
                    how="inner", suffixes=(None, "_atClose"))
    data["TimeIndex"] = data["Date"].astype(str) + " " + data["Time"]
    data["TimeIndex"] = pd.to_datetime(data["TimeIndex"])
    data = data.set_index("TimeIndex")
    
    ## Resampling with given number of minutes aggregations (AGGRE MIN)
    resample_data = pd.DataFrame()
    resample_data["High"] = data["High"].resample("{}T".format(aggre_min), origin="start_day").max()
    resample_data["Low"] = data["Low"].resample("{}T".format(aggre_min), origin="start_day").min()
    resample_data["Open"] = data["Open"].resample("{}T".format(aggre_min), origin="start_day").first()
    resample_data["Close"] = data["Close"].resample("{}T".format(aggre_min), origin="start_day").last()
    resample_data["Auction"] = data["Price"].resample("{}T".format(aggre_min), origin="start_day").last()
    resample_data = resample_data.dropna(how="all")
    
    resample_data["Volume"] = data["Volume"].resample("{}T".format(aggre_min), origin="start_day").sum()
    resample_data["Date"] = pd.to_datetime(resample_data.index.date)
    resample_data["RIC"] = stock_RIC
    
    ## Dropping days with abnormal trading minutes (# min != 390)
    day_check = target_data.groupby("Date")["Time"].count()
    abnormal_day = day_check[day_check != 390].index.values
    resample_data = resample_data.loc[~resample_data.Date.isin(abnormal_day)]
    
    resample_data.reset_index(inplace=True, drop=False)
    
    resample_data["Vol"] = abs(np.log(resample_data['Close']/resample_data['Open']))
    
    ## Shifting TimeIndex so hh:mm:ss represents the 10 min intervals prior of that.
    ## For example, TimeIndex 09:40:00 represents the time range 09:30:00 to 09:39:00
    resample_data["TimeIndex"] = resample_data["TimeIndex"] + pd.to_timedelta("{} minutes".format(aggre_min))
    
    ## Calculate daily average volatility measure
    sqrt_mse = lambda x: np.sqrt((x ** 2).sum())
    coef = np.sqrt(resample_data.groupby("Date")["Vol"].count())
    day_Vol = (resample_data.groupby("Date")["Vol"].apply(sqrt_mse) * coef).reset_index(drop=False)
    
    resample_data = pd.merge(resample_data, day_Vol, left_on="Date", right_on="Date", how="inner", suffixes=(None, "_Day"))

    ## Merging price at 3:50 information into the data
    data_350 = data.loc[(data.index.hour==15) & (data.index.minute==50) & (data.index.second==0)]
    full_data = pd.merge(resample_data, data_350[["Date", "RIC", "Open"]],
                 left_on=["Date", "RIC"],
                 right_on=["Date", "RIC"],
                 how="inner", suffixes=(None, "_at_3:50"))
    full_data["Auction_logdiff"] = abs(np.log(full_data["Auction"] / full_data["Open_at_3:50"]))
    if start_date != None:
        full_data = full_data.loc[full_data["TimeIndex"] >= start_time]
        day_Vol = day_Vol.loc[day_Vol["Date"] >= start_date]
    if end_date != None:
        full_data = full_data.loc[full_data["TimeIndex"] <= pd.to_datetime(end_time)]
        day_Vol = day_Vol.loc[day_Vol["Date"] <= end_date]
    
    full_data = full_data.reset_index(drop=True)
    
    return full_data, day_Vol
