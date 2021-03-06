import pathlib
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


def predict_single_var_future(df_data, header_name, forecast_periods):

    df_data.rename(columns={header_name: 'y'}, inplace=True)

    df_log = df_data.copy()
    df_log['y'] = np.log(df_log['y'])

    m = Prophet()
    m.fit(df_log)
    future = m.make_future_dataframe(periods=forecast_periods)
    forecast = m.predict(future)

    print(forecast.head())
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # fig1 = m.plot(forecast)
    # fig2 = m.plot_components(forecast)

    df_close = pd.DataFrame(df_data[['ds', 'y']]).set_index('ds')
    print(df_close)
    forecast_with_org_data = forecast.set_index('ds').join(df_close)
    print(forecast_with_org_data)
    forecast_with_org_data = forecast_with_org_data[['y', 'yhat', 'yhat_upper', 'yhat_lower']]
    forecast_with_org_data['yhat'] = np.exp(forecast_with_org_data.yhat)
    forecast_with_org_data['yhat_upper'] = np.exp(forecast_with_org_data.yhat_upper)
    forecast_with_org_data['yhat_lower'] = np.exp(forecast_with_org_data.yhat_lower)
    forecast_with_org_data[['y', 'yhat', 'yhat_upper', 'yhat_lower']].plot(figsize=(8, 6))
    print(forecast_with_org_data)
    # plt.show()
    forecast_with_org_data.rename(columns={'yhat': header_name}, inplace=True)
    return forecast_with_org_data[header_name][-1*forecast_periods:]


def main():
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_colwidth', None)

    root = pathlib.Path(__file__).parent.resolve()
    dataset_path = root / ".." / "dataset"
    stock_a = dataset_path / "stock" / "historical-quotes" / "A.json"
    print(root)

    df = get_prophet_data(stock_a)
    regressors = {}

    name = "poc"

    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    trainsize = int(df.shape[0] * 0.9)
    train = df[0:trainsize]
    test = df[trainsize:df.shape[0]]

    o = predict_single_var_future(df[['ds', 'Volume']].copy(), 'Volume', df.shape[0] - trainsize)
    regressors[name + '_' + 'Volume'] = pd.concat([df['Volume'], o], ignore_index=True)

    o2 = predict_single_var_future(df[['ds', 'Open']].copy(), 'Open', df.shape[0] - trainsize)
    regressors[name + '_' + 'Open'] = pd.concat([df['Open'], o2], ignore_index=True)

    o3 = predict_single_var_future(df[['ds', 'High']].copy(), 'High', df.shape[0] - trainsize)
    regressors[name + '_' + 'High'] = pd.concat([df['High'], o3], ignore_index=True)

    o4 = predict_single_var_future(df[['ds', 'Low']].copy(), 'Low', df.shape[0] - trainsize)
    regressors[name + '_' + 'Low'] = pd.concat([df['Low'], o4], ignore_index=True)

    df_log = df.copy()

    df_log['y'] = np.log(df_log['y'])
    df_log['Open'] = np.log(df_log['Open'])
    df_log['High'] = np.log(df_log['High'])
    df_log['Low'] = np.log(df_log['Low'])
    df_log['Volume'] = np.log(df_log['Volume'])

    df_log[name + '_' + 'Volume'] = df_log['Volume']
    df_log[name + '_' + 'Open'] = df_log['Open']
    df_log[name + '_' + 'High'] = df_log['High']
    df_log[name + '_' + 'Low'] = df_log['Low']

    m = Prophet()
    m.add_regressor(name + '_' + 'Volume')
    m.add_regressor(name + '_' + 'Open')
    m.add_regressor(name + '_' + 'High')
    m.add_regressor(name + '_' + 'Low')
    m.fit(df_log)
    future = m.make_future_dataframe(periods=df.shape[0] - trainsize)
    future[name + '_' + 'Volume'] = np.log(regressors[name + '_' + 'Volume'])
    future[name + '_' + 'Open'] = np.log(regressors[name + '_' + 'Open'])
    future[name + '_' + 'High'] = np.log(regressors[name + '_' + 'High'])
    future[name + '_' + 'Low'] = np.log(regressors[name + '_' + 'Low'])

    forecast = m.predict(future)
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

    plt.show()

    ae = (forecast_with_org_data_dropna['yhat'] - forecast_with_org_data_dropna['y'])
    print(ae.describe())

    print("MSE:",
          metrics.mean_squared_error(forecast_with_org_data_dropna['yhat'][trainsize:df.shape[0]], test['y']))
    print("MAE:",
          metrics.mean_absolute_error(forecast_with_org_data_dropna['yhat'][trainsize:df.shape[0]], test['y']))


if __name__ == "__main__":
    print('hello world')
    main()
