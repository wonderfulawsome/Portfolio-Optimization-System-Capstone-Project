# main.py
import os
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from prophet import Prophet
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# 랜덤 고정
np.random.seed(42)

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("FMP_API", "DEMO_KEY")

def get_stock_data(ticker, days=200):
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={API_KEY}"
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data['historical'][:days])
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    except:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
        close = np.random.normal(500, 50, days).cumsum() + 400
        volume = np.random.randint(2e7, 5e7, days)
        df = pd.DataFrame({'date': dates, 'close': close, 'volume': volume})
        return df.sort_values('date')

def add_technical_indicators(df):
    df = df.copy()
    df['sma20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['sma50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['sma200'] = SMAIndicator(close=df['close'], window=5).sma_indicator()
    df['volume_sma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma20']
    df['volume_spike'] = np.where(df['volume_ratio'] > 1.5, 1, 0)
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df = df.fillna(method='ffill').fillna(method='bfill').reset_index(drop=True)
    return df

@app.route("/forecast/<ticker>", methods=["GET"])
def forecast(ticker):
    df = get_stock_data(ticker, 200)
    df = add_technical_indicators(df)

    train_df = df[['date','close','sma20','sma50','volume_ratio','rsi']].rename(columns={'date':'ds','close':'y'})
    train_df = train_df.fillna(method='ffill').fillna(method='bfill')

    model = Prophet(
        changepoint_prior_scale=0.1,
        seasonality_mode='multiplicative',
        yearly_seasonality=5,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.add_regressor('sma20', standardize=False)
    model.add_regressor('sma50', standardize=False)
    model.add_regressor('volume_ratio', standardize=True)
    model.add_regressor('rsi', standardize=True)
    model.fit(train_df)

    future = model.make_future_dataframe(periods=30)
    last_date = df['date'].max()
    last_sma20 = df.loc[df['date']==last_date,'sma20'].values[0]
    last_sma50 = df.loc[df['date']==last_date,'sma50'].values[0]
    last_rsi   = df.loc[df['date']==last_date,'rsi'].values[0]

    for col in ['sma20','sma50','volume_ratio','rsi']:
        for d, val in zip(df['date'], df[col]):
            future.loc[future['ds']==d, col] = val

    future_dates = future[future['ds'] > last_date]['ds']
    for i, d in enumerate(future_dates):
        future.loc[future['ds']==d,'sma20'] = last_sma20*(1+np.random.normal(0,0.002)*(i+1))
        future.loc[future['ds']==d,'sma50'] = last_sma50*(1+np.random.normal(0,0.001)*(i+1))
        if i==0:
            future.loc[future['ds']==d,'rsi'] = last_rsi
        else:
            prev_rsi = future.loc[future['ds']==future_dates.iloc[i-1],'rsi'].values[0]
            new_rsi = prev_rsi + np.random.normal(0, 1.5)
            if new_rsi>70: new_rsi -= np.random.uniform(1, 2.5)
            if new_rsi<30: new_rsi += np.random.uniform(1, 2.5)
            future.loc[future['ds']==d,'rsi'] = max(0, min(100, new_rsi))
        future.loc[future['ds']==d,'volume_ratio'] = max(0.5, np.random.normal(1, 0.1))

    future = future.fillna(method='ffill')
    forecast_df = model.predict(future)

    for i in range(1, len(forecast_df)):
        if forecast_df['ds'].iloc[i] <= last_date: 
            continue
        base_vol = np.random.normal(0, 0.005)
        rsi_val  = future.loc[i,'rsi']
        if rsi_val > 70:  rsi_eff = np.random.uniform(-0.01, -0.002)
        elif rsi_val <30: rsi_eff = np.random.uniform(0.002, 0.01)
        else:             rsi_eff = 0
        vol_eff=0
        if future.loc[i,'volume_ratio']>1.5:
            vol_eff = np.random.uniform(-0.01, 0.01)
        price = forecast_df.loc[i,'yhat']
        sma20 = future.loc[i,'sma20']
        sma_eff = 0
        if price < sma20*0.98:
            sma_eff = np.random.uniform(0.002, 0.01)
        elif price > sma20*1.02:
            sma_eff = np.random.uniform(-0.01, -0.002)
        total_eff = base_vol + rsi_eff + vol_eff + sma_eff
        forecast_df.loc[i,'yhat'] *= (1+ total_eff)
        forecast_df.loc[i,'yhat_lower'] = forecast_df.loc[i,'yhat']*0.97
        forecast_df.loc[i,'yhat_upper'] = forecast_df.loc[i,'yhat']*1.03

    current_price = df['close'].iloc[-1]
    sma_now = [df['sma20'].iloc[-1], df['sma50'].iloc[-1], df['sma200'].iloc[-1]]
    support_list = [x for x in sma_now if x < current_price]
    resist_list  = [x for x in sma_now if x > current_price]

    window = 10
    for i in range(window, len(df)-window):
        if df['close'].iloc[i] == max(df['close'].iloc[i-window:i+window]):
            if df['close'].iloc[i] > current_price:
                resist_list.append(df['close'].iloc[i])
        if df['close'].iloc[i] == min(df['close'].iloc[i-window:i+window]):
            if df['close'].iloc[i] < current_price:
                support_list.append(df['close'].iloc[i])

    volume_spike_dates = df.loc[df['volume_spike']==1,'date'].dt.strftime("%Y-%m-%d").tolist()

    real_rows = df[['date','close','sma20','sma50','sma200','volume_spike','rsi']]
    pred_rows = forecast_df[forecast_df['ds']>last_date].copy()

    real_data = []
    for _, row in real_rows.iterrows():
        real_data.append({
            "ds": row['date'].strftime("%Y-%m-%d"),
            "close": row['close'],
            "sma20": row['sma20'],
            "sma50": row['sma50'],
            "sma200": row['sma200'],
            "volume_spike": int(row['volume_spike']),
            "rsi": row['rsi']
        })

    pred_data = []
    for _, row in pred_rows.iterrows():
        dstr = row['ds'].strftime("%Y-%m-%d")
        sma20_val = future.loc[future['ds']==row['ds'],'sma20'].values[0]
        sma50_val = future.loc[future['ds']==row['ds'],'sma50'].values[0]
        vol_spike = 1 if future.loc[future['ds']==row['ds'],'volume_ratio'].values[0]>1.5 else 0
        rsi_val   = future.loc[future['ds']==row['ds'],'rsi'].values[0]
        pred_data.append({
            "ds": dstr,
            "yhat": row['yhat'],
            "yhat_lower": row['yhat_lower'],
            "yhat_upper": row['yhat_upper'],
            "sma20": sma20_val,
            "sma50": sma50_val,
            "volume_spike": vol_spike,
            "rsi": rsi_val
        })

    return jsonify({
        "real": real_data,
        "predicted": pred_data,
        "support": sorted(support_list),
        "resistance": sorted(resist_list),
        "forecastStart": last_date.strftime("%Y-%m-%d"),
        "volumeSpikes": volume_spike_dates
    })

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
