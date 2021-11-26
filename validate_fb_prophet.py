import pathlib
import logging
import pandas as pd

from models.fb_prophet import LibFBProphet

train_ratio = 0.9


def prepare_model_data_for_stock(stock_path, name):
    with open(stock_path, 'r', encoding='utf-8') as f:
        data = {
            'args': {
                'using_regressors': ['Open', 'High', 'Low', 'Volume'],
                'train_ratio': train_ratio,
                'enable_plot': True
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

    model = LibFBProphet()

    # do fb_prophet forecast for single stock data
    symbol = 'AAPL'
    stock_path = stock_historical_path / (symbol + ".json")
    stock_data = prepare_model_data_for_stock(stock_path, symbol)
    logging.debug(stock_data)

    model.run_validate(stock_data)


if __name__ == "__main__":
    main()
