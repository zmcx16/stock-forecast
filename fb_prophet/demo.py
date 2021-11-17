import os
import sys
import pathlib
import json
import requests
import csv
import time
from urllib.parse import urlencode
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn import metrics


def get_prophet_data(stock_path):
    with open(stock_path, 'r', encoding='utf-8') as f:
        df = pd.read_json(f.read(), orient='records')
        print(df)

    # rename
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    print(df)
    return df


def main():
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_colwidth', None)

    root = pathlib.Path(__file__).parent.resolve()
    dataset_path = root / ".." / "dataset"
    stock_a = dataset_path / "stock" / "historical-quotes" / "A.json"
    print(root)

    df = get_prophet_data(stock_a)
    df_log = df.copy()

    df_log['y'] = np.log(df_log['y'])
    df_log['Open'] = np.log(df_log['Open'])
    df_log['High'] = np.log(df_log['High'])
    df_log['Low'] = np.log(df_log['Low'])
    df_log['Volume'] = np.log(df_log['Volume'])

    m = Prophet()
    m.fit(df_log)
    future = m.make_future_dataframe(periods=14)
    future.tail()

    forecast = m.predict(future)
    print(forecast.head())
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)

    df_close = pd.DataFrame(df[['ds', 'y']]).set_index('ds')
    print(df_close)
    forecast_with_org_data = forecast.set_index('ds').join(df_close)
    print(forecast_with_org_data)
    forecast_with_org_data = forecast_with_org_data[['y', 'yhat', 'yhat_upper', 'yhat_lower']]
    forecast_with_org_data['yhat'] = np.exp(forecast_with_org_data.yhat)
    forecast_with_org_data['yhat_upper'] = np.exp(forecast_with_org_data.yhat_upper)
    forecast_with_org_data['yhat_lower'] = np.exp(forecast_with_org_data.yhat_lower)
    forecast_with_org_data[['y', 'yhat', 'yhat_upper', 'yhat_lower']].plot(figsize=(8, 6))
    print(forecast_with_org_data)

    forecast_with_org_data_dropna = forecast_with_org_data.dropna()
    forecast_with_org_data_dropna[['y', 'yhat']].plot(figsize=(8, 6))

    # plt.show()

    ae = (forecast_with_org_data_dropna['yhat'] - forecast_with_org_data_dropna['y'])
    print(ae.describe())

    print("MSE:", metrics.mean_squared_error(forecast_with_org_data_dropna['yhat'], forecast_with_org_data_dropna['y']))
    print("MAE:", metrics.mean_absolute_error(forecast_with_org_data_dropna['yhat'], forecast_with_org_data_dropna['y']))


if __name__ == "__main__":
    print('hello world')
    main()
