import os
import pathlib
import json
import logging

import pandas as pd
from datetime import datetime
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
    logging.basicConfig(level=logging.INFO)

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

    output_table = {'update_time': str(datetime.now()), 'data': {}}
    # do fb_prophet forecast for single stock data
    for symbol in symbol_list:
        logging.info('{} stock forecast start'.format(symbol))
        stock_path = stock_historical_path / (symbol + '.json')
        stock_data = prepare_model_data_for_stock(stock_path, symbol)
        logging.debug(stock_data)

        forecast = model.run_predict(stock_data)
        logging.debug(forecast)
        forecast['Date'] = forecast['Date'].apply(lambda x: x.strftime('%m/%d/%Y'))
        forecast = forecast[::-1]  # reverse to latest order
        forecast_json = forecast.to_dict(orient='records')
        logging.debug(forecast_json)

        fcst = {'FCST': '-', 'FCST_Upper' + str(forecast_periods): '-',
                                            'FCST_Lower' + str(forecast_periods): '-'}
        if len(forecast_json) > 0:
            fcst['FCST'] = round(forecast_json[0]['Predict'], 3)
            fcst['FCST_Upper' + str(forecast_periods)] = round(forecast_json[0]['Predict_Upper'], 3)
            fcst['FCST_Lower' + str(forecast_periods)] = round(forecast_json[0]['Predict_Lower'], 3)

        output_table['data'][symbol] = fcst

        with open(stock_fbprophet_ohlv_path / (symbol + '.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(forecast_json, separators=(',', ':')).replace('NaN', '"-"'))

        logging.info('{} stock forecast done'.format(symbol))

    with open(output_path / 'stock_fbprophet_ohlv.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(output_table, separators=(',', ':')))

    logging.info('main done')


if __name__ == "__main__":
    main()
