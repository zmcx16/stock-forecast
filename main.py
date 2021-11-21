import os
import sys
import base64
import pathlib
import json
import requests
import csv
import threading
import queue
import time
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn import metrics
from urllib.parse import urlencode
from datetime import datetime, timedelta

from models.fb_prophet import LibFBProphet

forecast_periods = 30


def get_all_stock_symbol(stat_path):
    symbol_list = []
    with open(stat_path, 'r', encoding='utf-8') as f:
        stat = json.loads(f.read())
        for symbol in stat:
            symbol_list.append(symbol)

    return symbol_list


def prepare_model_data_for_stock(stock_path, name):
    with open(stock_path, 'r', encoding='utf-8') as f:
        data = {
            'args': {
                'using_regressors': ['Open', 'High', 'Low', 'Volume'],
                'forecast_periods': forecast_periods
            },
            'target_data': {
                'name': name,
                'data': pd.read_json(f.read(), orient='records'),
                'type': 'stock'
            },
            'feature_data': []
        }
        return data


def main():
    pd.set_option('display.max_columns', None)
    logging.basicConfig(level=logging.DEBUG)

    root_path = pathlib.Path(__file__).parent.resolve()
    dataset_path = root_path / "dataset"
    stock_folder_path = dataset_path / "stock"
    stock_stat_path = stock_folder_path / 'stat.json'
    stock_historical_path = stock_folder_path / "historical-quotes"

    # output
    output_path = root_path / "forecast_output"
    stock_fbprophet_ohlv_path = output_path / "stock_fbprophet_ohlv"
    if not os.path.exists(stock_fbprophet_ohlv_path):
        os.makedirs(stock_fbprophet_ohlv_path)

    model = LibFBProphet()

    symbol_list = get_all_stock_symbol(stock_stat_path)
    logging.info(symbol_list)

    output_table = {'update_time': str(datetime.now()), "data": {}}
    # do fb_prophet forecast for single stock data
    for symbol in symbol_list:
        stock_path = stock_historical_path / ('T' + ".json")
        stock_data = prepare_model_data_for_stock(stock_path, symbol)
        logging.debug(stock_data)

        forecast = model.run_predict(stock_data)
        logging.debug(forecast)
        forecast['Date'] = forecast['Date'].apply(lambda x: x.strftime('%m/%d/%Y'))
        forecast = forecast[::-1]  # reverse to latest order
        forecast_json = forecast.to_dict(orient='records')
        logging.debug(forecast_json)

        with open(stock_fbprophet_ohlv_path / (symbol + '.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(forecast_json, separators=(',', ':')))

        break


if __name__ == "__main__":
    main()
