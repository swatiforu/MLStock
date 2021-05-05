from firebase import firebase
import pandas as pd
import numpy as np
import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, redirect, url_for, flash, jsonify

app = Flask(__name__)

fb = firebase.FirebaseApplication('https://historicaldatafyp-default-rtdb.firebaseio.com/', None)

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
