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

max_thread = 64
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


class FBProphetThread(threading.Thread):

    def __init__(self, id, output_path, task_queue, gcp_url):
        threading.Thread.__init__(self)
        self.id = id
        self.output_path = output_path
        self.task_queue = task_queue
        self.output_table = {}
        self.gcp_url = gcp_url

    @staticmethod
    def run_fb_prophet(model_input):
        model = LibFBProphet()
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
                model_input = data["model_input"]
                task_type = model_input["type"]
                if task_type not in self.output_table:
                    self.output_table[task_type] = {}

                forecast_periods = model_input["args"]["forecast_periods"]

                output_folder = self.output_path / task_type
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                logging.info('{} stock forecast start'.format(symbol))
                if self.gcp_url != '':
                    ret, forecast_json = send_post_json(gcp_url, json.dumps(model_input))
                    if ret != 0:
                        logging.error('send_post_json failed: {ret}'.format(ret=ret))
                        continue
                else:
                    forecast_json = FBProphetThread.run_fb_prophet(model_input)

                logging.debug(forecast_json)

                fcst = {'FCST': '-', 'FCST_Upper' + str(forecast_periods): '-',
                        'FCST_Lower' + str(forecast_periods): '-'}
                if len(forecast_json) > 0:
                    fcst['FCST'] = round(forecast_json[0]['Predict'], 3)
                    fcst['FCST_Upper' + str(forecast_periods)] = round(forecast_json[0]['Predict_Upper'], 3)
                    fcst['FCST_Lower' + str(forecast_periods)] = round(forecast_json[0]['Predict_Lower'], 3)

                self.output_table[task_type][symbol] = fcst

                with open(output_folder / (symbol + '.json'), 'w', encoding='utf-8') as f:
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
                    json.dumps(request_json['model_input']))).replace('NaN', '"-"')
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
    forecast_config_path = root_path / "forecast_config.json"

    # output
    output_path = root_path / "forecast_output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    task_queue = queue.Queue()

    with open(forecast_config_path, 'r', encoding='utf-8') as f:
        c = f.read()
        logging.info(c)
        forecast_config = json.loads(c)
        for symbol in forecast_config:
            for model_input in forecast_config[symbol]:
                model_input["target_data"]["file_path"] = \
                    dataset_path / model_input["target_data"]["file_path"]

                task_queue.put({"symbol": symbol, "model_input": model_input})

    work_list = []
    for index in range(max_thread):
        work_list.append(FBProphetThread(index, output_path, task_queue, gcp_url))
        work_list[index].start()

    for worker in work_list:
        worker.join()

    # save output_table
    output_table = {}
    for worker in work_list:
        for t in worker.output_table:
            for k in worker.output_table[t]:
                if t not in output_table:
                    output_table[t] = {'update_time': str(datetime.now()), 'data': {}}

                output_table[t]['data'][k] = worker.output_table[t][k]

    for t in output_table:
        with open(output_path / (t + '.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(output_table[t], separators=(',', ':')))

    logging.info('all task done')


if __name__ == "__main__":
    main()
