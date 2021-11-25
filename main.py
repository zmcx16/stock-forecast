import os
import pathlib
import json
import logging
import threading
import requests
import queue

import pandas as pd
from datetime import datetime
from models.fb_prophet import LibFBProphet

forecast_periods = 30
max_thread = 16
gcp_url = os.environ.get("GCP_URL", "")


def send_post_json(url, req_data):
    try:
        headers = {'content-type': 'application/json'}
        res = requests.post(url, req_data, headers=headers)
        res.raise_for_status()
    except Exception as ex:
        logging.error('Generated an exception: {ex}'.format(ex=ex))
        return -1, ex

    return 0, res.json()


def get_all_stock_symbol(stat_path):
    symbol_list = []
    with open(stat_path, 'r', encoding='utf-8') as f:
        stat = json.loads(f.read())
        for symbol in stat:
            symbol_list.append(symbol)

    return symbol_list


def prepare_model_data_for_stock(stock_data, name):
    return {
        'args': {
            'using_regressors': ['Open', 'High', 'Low', 'Volume'],
            'forecast_periods': forecast_periods
        },
        'target_data': {
            'name': name,
            'data': pd.read_json(stock_data, orient='records'),
            'type': 'stock'
        },
        'feature_data': []
    }


class FBProphetThread(threading.Thread):

    def __init__(self, id, stock_historical_path, stock_fbprophet_ohlv_path, task_queue, gcp_url):
        threading.Thread.__init__(self)
        self.id = id
        self.stock_historical_path = stock_historical_path
        self.stock_fbprophet_ohlv_path = stock_fbprophet_ohlv_path
        self.task_queue = task_queue
        self.output_table = {}
        self.gcp_url = gcp_url

    @staticmethod
    def run_fb_prophet(stock_data, name):
        model = LibFBProphet()
        model_input = prepare_model_data_for_stock(stock_data, name)
        logging.debug(model_input)
        forecast = model.run_predict(model_input)
        logging.debug(forecast)
        forecast['Date'] = forecast['Date'].apply(lambda x: x.strftime('%m/%d/%Y'))
        forecast = forecast[::-1]  # reverse to latest order
        return forecast.to_dict(orient='records')

    def run(self):
        logging.info("Thread{} start".format(self.id))
        while self.task_queue.qsize() > 0:
            try:
                data = self.task_queue.get()
                symbol = data["symbol"]

                logging.info('{} stock forecast start'.format(symbol))
                stock_path = self.stock_historical_path / (symbol + '.json')
                with open(stock_path, 'r', encoding='utf-8') as f:
                    d = f.read()
                    if self.gcp_url != '':
                        gcp_api_input = {'name': symbol, 'stock_data': json.loads(d)}
                        ret, forecast_json = send_post_json(gcp_url, json.dumps(gcp_api_input))
                        if ret != 0:
                            logging.error('send_post_json failed: {ret}'.format(ret=ret))
                            continue
                    else:
                        forecast_json = FBProphetThread.run_fb_prophet(d, symbol)

                logging.debug(forecast_json)

                fcst = {'FCST': '-', 'FCST_Upper' + str(forecast_periods): '-',
                        'FCST_Lower' + str(forecast_periods): '-'}
                if len(forecast_json) > 0:
                    fcst['FCST'] = round(forecast_json[0]['Predict'], 3)
                    fcst['FCST_Upper' + str(forecast_periods)] = round(forecast_json[0]['Predict_Upper'], 3)
                    fcst['FCST_Lower' + str(forecast_periods)] = round(forecast_json[0]['Predict_Lower'], 3)

                self.output_table[symbol] = fcst

                with open(self.stock_fbprophet_ohlv_path / (symbol + '.json'), 'w', encoding='utf-8') as f:
                    f.write(json.dumps(forecast_json, separators=(',', ':')).replace('NaN', '"-"'))

                logging.info('{} stock forecast done'.format(symbol))

            except Exception as ex:
                logging.error('Generated an exception: {ex}'.format(ex=ex))

        logging.info("Thread{} end".format(self.id))


def gcp_api_main(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    logging.basicConfig(level=logging.INFO)
    try:
        request_json = request.get_json()
        if request.args and 'message' in request.args:
            return request.args.get('message')
        elif request_json and 'message' in request_json:
            return request_json['message']
        elif request_json and 'stock_data' in request_json and 'name' in request_json:
            logging.info('run_fb_prophet')
            return json.dumps(
                FBProphetThread.run_fb_prophet(
                    json.dumps(request_json['stock_data']), request_json['name'])).replace('NaN', '"-"')
        else:
            return f'Hello World!'

    except Exception as ex:
        err_msg = 'Generated an exception: {ex}'.format(ex=ex)
        logging.error(err_msg)
        return err_msg


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

    symbol_list = get_all_stock_symbol(stock_stat_path)
    logging.info(symbol_list)

    task_queue = queue.Queue()

    for symbol in symbol_list:
        data = {"symbol": symbol}
        task_queue.put(data)

    work_list = []
    for index in range(max_thread):
        work_list.append(FBProphetThread(index, stock_historical_path, stock_fbprophet_ohlv_path, task_queue, gcp_url))
        work_list[index].start()

    for worker in work_list:
        worker.join()

    # save output_table
    output_table = {'update_time': str(datetime.now()), 'data': {}}
    for worker in work_list:
        for k in worker.output_table:
            output_table['data'][k] = worker.output_table[k]

    with open(output_path / 'stock_fbprophet_ohlv.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(output_table, separators=(',', ':')))

    logging.info('all task done')


if __name__ == "__main__":
    main()
