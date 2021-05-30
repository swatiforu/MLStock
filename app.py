from firebase import firebase
import pandas as pd
import numpy as np
import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

fb = firebase.FirebaseApplication('https://historicaldatafyp-default-rtdb.firebaseio.com/', None)
@app.route('/technical/', methods=['GET'])
def getTechnical():
  ticker = request.args['Ticker']
  a = fb.get('historicaldatafyp-default-rtdb/Stocks/'+ticker, '')
  df = pd.DataFrame(a)
  df['Volume'] = [int(x.replace(',', '')) for x in df.Volume.values.copy()]
  df['Close'] = pd.to_numeric(df['Close'])
  values = dict()
  values['OBV'] = calculate_OBV(df)
  values['MACD'] = calculate_MACD(df)
  values['RSI'] = calculate_RSI(df)
  return values
def calculate_OBV(data):
  obv = []
  obv.append(0)
  for i in range(1, len(data.Close.values)):
    if data.Close[i]>data.Close[i-1]:
      obv.append(obv[-1]+data.Volume[i])
    elif data.Close[i]<data.Close[i-1]:
      obv.append(obv[-1]-data.Volume[i])
    else:
      obv.append(obv[-1])
  new_df = pd.DataFrame(index = data.index)
  new_df['OBV'] = obv
  new_df['OBV_EMA'] = new_df['OBV'].ewm(span = 20).mean()
  val = new_df.values[-1]
  if val[0] >= val[1]:
    return "Buyer Pressure"
  else:
    return "Seller Pressure"
def calculate_MACD(data):
  ShortEMA = data.Close.ewm(span=12, adjust=False).mean()
  LongEMA = data.Close.ewm(span=26, adjust=False).mean()
  MACD = ShortEMA - LongEMA
  signal = MACD.ewm(span=9, adjust=False).mean()
  new_df = pd.DataFrame()
  new_df['MACD'] = MACD
  new_df['Signal'] = signal
  val = new_df.values[-1]
  if val[0]>val[1]:
    return "Buy Signal"
  else:
    return "Sell Signal"
def calculate_RSI (data):
  time_window = 14
  diff = data.Close.diff(1).dropna()
  up_chg = 0 * diff
  down_chg = 0 * diff
  up_chg[diff > 0] = diff[ diff>0 ]
  down_chg[diff < 0] = diff[ diff < 0 ]
  up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
  down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
  rs = abs(up_chg_avg/down_chg_avg)
  rsi = 100 - 100/(1+rs)
  if rsi.values[-1]>=70:
    return "Overbought"
  elif rsi.values[-1]<=30:
    return "Oversold"
  else:
    return "Normal"  

@app.route('/predictions/', methods=['GET'])
def getPredictions():
  ticker = request.args['Ticker']
  return prediction7Days(ticker)

def get7dates(date):
  lastdate = datetime.datetime.strptime(date, '%d-%B-%Y')
  all_dates = []
  for i in range(1,8):
    d = lastdate + datetime.timedelta(days = i)
    all_dates.append(d.strftime('%d-%B-%Y'))
  return all_dates

def prediction7Days(ticker):
  model = load_model('Weights FYP/'+ticker+'_weights.h5')
  a = fb.get('historicaldatafyp-default-rtdb/Stocks/'+ticker, '')
  df = pd.DataFrame(a)
  all_dates = get7dates(df['Date'].tail(1).values[0])
  df.set_index(pd.to_datetime(df['Date']), inplace = True)
  df.drop(['Date'], axis = 1, inplace = True)
  df['Close'] =pd.to_numeric(df['Close'])
  data = df.filter(['Close'])
  dataset = data.values
  window = 7
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)
  values = scaled_data[-window:,:]
  predicted = []
  vals = []
  prev_bloc = values.copy()
  for i in range(window):
    vals.append(np.asarray([prev_bloc]))
    p = model.predict(np.asarray([prev_bloc]))
    for j in range(len(prev_bloc)-1):
      prev_bloc[j] = prev_bloc[j+1]
    prev_bloc[-1] = p[0]
    predicted.append(scaler.inverse_transform(p)[0][0])
    prev_bloc = np.asarray(prev_bloc)
  predicted = np.asarray(predicted)
  predicted = list(predicted)
  result = {}
  for i in range(7):
    result[all_dates[i]] = str(predicted[i])
  return result
