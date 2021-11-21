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

train_ratio = 0.9


def prepare_model_data_for_stock(stock_path, name):
    with open(stock_path, 'r', encoding='utf-8') as f:
        data = {
            'args': {
                'using_regressors': ['Open', 'High', 'Low', 'Volume'],
                'train_ratio': train_ratio
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

    logging.basicConfig(level=logging.DEBUG)

    root_path = pathlib.Path(__file__).parent.resolve()
    dataset_path = root_path / "dataset"
    stock_folder_path = dataset_path / "stock"
    stock_stat_path = stock_folder_path / 'stat.json'
    stock_historical_path = stock_folder_path / "historical-quotes"

    model = LibFBProphet()

    # do fb_prophet forecast for single stock data
    symbol = 'T'
    stock_path = stock_historical_path / (symbol + ".json")
    stock_data = prepare_model_data_for_stock(stock_path, symbol)
    logging.debug(stock_data)

    model.run_validate(stock_data)


if __name__ == "__main__":
    main()
