import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn import metrics

from models.model_abc import Model


# https://github.com/facebook/prophet/issues/223
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class LibFBProphet(Model):

    enable_plot = False
    train_ratio = 0.9

    def __init__(self):
        logging.getLogger('fbprophet').setLevel(logging.WARNING)

    # data = {
    #       'args': {
    #           'using_regressors': ['Open', 'High', 'Low', 'Volume']
    #           'forecast_periods': 30 # use for run_predict
    #           'training_ratio': 0.9 # use for run_validate
    #       }
    #       'target_data': {
    #           'name': 'name'
    #           'data': obj      # dataframe
    #           'type': 'stock'  # stock or market
    #       },
    #       'feature_data': [
    #           {
    #               'name': 'name'
    #               'data': obj      # dataframe
    #               'type': 'stock'  # stock or market
    #           }
    #       ]
    #   }

    def run_validate(self, data):

        logging.debug(data['args'])
        if 'enable_plot' in data['args']:
            self.enable_plot = data['args']['enable_plot']
        if 'train_ratio' in data['args']:
            self.train_ratio = data['args']['train_ratio']

        using_regressors = data['args']['using_regressors']
        name = data['target_data']['name']

        # reverse data order from latest start -> oldest start
        df = data['target_data']['data'][::-1]

        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

        train_size = int(df.shape[0] * self.train_ratio)
        train_data = df[0:train_size]
        test_data = df[train_size:df.shape[0]]

        forecast_with_org_data = self.__run_model(train_data, using_regressors, df.shape[0] - train_size, name)

        if self.enable_plot:
            plt.show()

        logging.info("MSE: {}".format(
              metrics.mean_squared_error(forecast_with_org_data['yhat'][train_size:df.shape[0]], test_data['y'])))
        logging.info("MAE: {}".format(
              metrics.mean_absolute_error(forecast_with_org_data['yhat'][train_size:df.shape[0]], test_data['y'])))

        return NotImplemented

    def run_predict(self, data):

        logging.debug(data['args'])
        if 'enable_plot' in data['args']:
            self.enable_plot = data['args']['enable_plot']

        using_regressors = data['args']['using_regressors']
        forecast_periods = data['args']['forecast_periods']
        name = data['target_data']['name']
        df = data['target_data']['data']

        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

        forecast_with_org_data = self.__run_model(df, using_regressors, forecast_periods, name)

        if self.enable_plot:
            plt.show()

        # rename
        final_forecast = forecast_with_org_data.reset_index()
        final_forecast.rename(
            columns={'ds': 'Date', 'y': 'Close',
                     'yhat': 'Predict', 'yhat_upper': 'Predict_Upper', 'yhat_lower': 'Predict_Lower',
                     'trend': 'Trend', 'trend_upper': 'Trend_Upper', 'trend_lower': 'Trend_Lower'}, inplace=True)
        return final_forecast

    def __run_model(self, df_data, using_regressors, forecast_periods, name):

        m = Prophet()

        df_log = df_data.copy()
        df_log['y'] = np.log(df_data['y'])

        regressors = {}
        for r in using_regressors:
            if r in df_data.columns.values:
                o = LibFBProphet.__predict_single_var_future(df_data[['ds', r]].copy(), r, forecast_periods)
                regressors[name + '_' + r] = pd.concat([df_data[r], o], ignore_index=True)
                df_log[name + '_' + r] = np.log(df_data[r])
                m.add_regressor(name + '_' + r)

        with suppress_stdout_stderr():
            m.fit(df_log)
        future = m.make_future_dataframe(periods=forecast_periods)

        for r in using_regressors:
            if r in df_data.columns.values:
                future[name + '_' + r] = np.log(regressors[name + '_' + r])

        forecast = m.predict(future)

        logging.debug(forecast)
        logging.debug(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        train_close = pd.DataFrame(df_data[['ds', 'y']]).set_index('ds')
        forecast_with_org_data = forecast.set_index('ds').join(train_close)

        forecast_with_org_data = forecast_with_org_data[['y', 'yhat', 'yhat_upper', 'yhat_lower', 'trend', 'trend_upper', 'trend_lower']]
        forecast_with_org_data['yhat'] = np.exp(forecast_with_org_data.yhat)
        forecast_with_org_data['yhat_upper'] = np.exp(forecast_with_org_data.yhat_upper)
        forecast_with_org_data['yhat_lower'] = np.exp(forecast_with_org_data.yhat_lower)

        if self.enable_plot:
            m.plot(forecast)
            m.plot_components(forecast)
            forecast_with_org_data[['y', 'yhat', 'yhat_upper', 'yhat_lower']].plot(figsize=(8, 6))

        return forecast_with_org_data

    @staticmethod
    def __predict_single_var_future(df_data, header_name, forecast_periods):

        df_data.rename(columns={header_name: 'y'}, inplace=True)

        df_log = df_data.copy()
        df_log['y'] = np.log(df_data['y'])

        m = Prophet()
        with suppress_stdout_stderr():
            m.fit(df_log)
        future = m.make_future_dataframe(periods=forecast_periods)
        forecast = m.predict(future)

        logging.debug(forecast.head())
        logging.debug(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        df_close = pd.DataFrame(df_data[['ds', 'y']]).set_index('ds')
        logging.debug(df_close)
        forecast_with_org_data = forecast.set_index('ds').join(df_close)
        logging.debug(forecast_with_org_data)
        forecast_with_org_data = forecast_with_org_data[['y', 'yhat', 'yhat_upper', 'yhat_lower']]
        forecast_with_org_data['yhat'] = np.exp(forecast_with_org_data.yhat)
        forecast_with_org_data['yhat_upper'] = np.exp(forecast_with_org_data.yhat_upper)
        forecast_with_org_data['yhat_lower'] = np.exp(forecast_with_org_data.yhat_lower)
        # forecast_with_org_data[['y', 'yhat', 'yhat_upper', 'yhat_lower']].plot(figsize=(8, 6))
        logging.debug(forecast_with_org_data)

        forecast_with_org_data.rename(columns={'yhat': header_name}, inplace=True)
        return forecast_with_org_data[header_name][-1*forecast_periods:]
