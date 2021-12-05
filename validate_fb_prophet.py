import pathlib
import logging
import pandas as pd

from models.fb_prophet import LibFBProphet

train_ratio = 0.9


def prepare_model_data_for_stock(root_path):
    return {
        'args': {
            'using_regressors': [],
            'train_ratio': train_ratio,
            'enable_plot': True
        },
        'target_data': {
            'name': "DAC",
            'file_path': root_path / "dataset/stock/historical-quotes/DAC.json",
            'type': 'stock'
        },
        'feature_data': [
            {
                'using_regressors': ["Close"],
                'name': "BDIY_IND",
                'file_path': root_path / "dataset/markets/historical-quotes/bloomberg_BDIY_IND.json",
                'type': 'market'
            },
            {
                'using_regressors': ["Close"],
                'name': "FBX",
                'file_path': root_path / "dataset/markets/historical-quotes/freightos_FBX.json",
                'type': 'market'
            }
        ]
    }


def main():
    pd.set_option('display.max_columns', None)
    logging.basicConfig(level=logging.DEBUG)

    root_path = pathlib.Path(__file__).parent.resolve()
    model = LibFBProphet()

    # do fb_prophet forecast for single stock data
    stock_data = prepare_model_data_for_stock(root_path)
    logging.debug(stock_data)
    model.run_validate(stock_data)


if __name__ == "__main__":
    main()
