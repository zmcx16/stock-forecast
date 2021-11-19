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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn import metrics
from urllib.parse import urlencode
from datetime import datetime, timedelta

from models.fb_prophet import LibFBProphet


def get_all_stock_symbol(stat_path):
    symbol_list = []
    with open(stat_path, 'r', encoding='utf-8') as f:
        stat = json.loads(f.read())
        for symbol in stat:
            symbol_list.append(symbol)

    return symbol_list


def prepare_model_data_for_stock(stock_path):
    with open(stock_path, 'r', encoding='utf-8') as f:
        data = {
            'target_data': {
                'name': 'name',
                'data': pd.read_json(f.read(), orient='records'),
                'type': 'stock'
            },
            'feature_data': []
        }
        return data


def main():
    root_path = pathlib.Path(__file__).parent.resolve()
    dataset_path = root_path / "dataset"
    stock_folder_path = dataset_path / "stock"
    stock_stat_path = stock_folder_path / 'stat.json'
    stock_historical_path = stock_folder_path / "historical-quotes"

    model = LibFBProphet()

    symbol_list = get_all_stock_symbol(stock_stat_path)
    print(symbol_list)

    # do fb_prophet forecast for single stock data
    for symbol in symbol_list:
        stock_path = stock_historical_path / ('T' + ".json")
        stock_data = prepare_model_data_for_stock(stock_path)
        print(stock_data)
        break

    model.run(stock_data)


if __name__ == "__main__":
    main()
