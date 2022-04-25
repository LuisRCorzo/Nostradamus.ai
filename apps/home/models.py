# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from datetime import datetime

import json

from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, event,exc
from collections import defaultdict

import pandas as pd


from apps import db,scheduler

import requests



class BTC(db.Model):

    
    __tablename__ = 'btc'
    time = db.Column(db.Float(), primary_key=True)
    high = db.Column(db.Float())
    low = db.Column(db.Float())
    open = db.Column(db.Float())
    close = db.Column(db.Float())
    volumeto = db.Column(db.Float())
    volumefrom = db.Column(db.Float())
    
    @classmethod
    def get_historical(self):
        return self.query.all()
    
    @classmethod
    def get_by_time(cls, time):
        return cls.query.filter_by(time=time).first()
    

    def toJSON(self):
        return self.toDICT()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

class BTC_forecasts(db.Model):

    __tablename__ = 'btc_predictions'
    #time = db.Column(db.Float(), primary_key=True)
    close = db.Column(db.Float(), primary_key=True)

    @classmethod
    def get_historical(self):
        return self.query.all()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

    def toJSON(self):
        return self.toDICT()
    
    def save(self):
        db.session.add(self)
        db.session.commit()


@event.listens_for(BTC.__table__, 'after_create')
def create_BTC(*args, **kwargs):

    api_key='api_key={ea0232c4ea8a3007655f1518de6af8ea6c4a5e546ddf83988ec885db9600a11e}'
    btcUrl='https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&allData=true&'
    resBTC = requests.get(btcUrl+api_key).json()['Data']['Data']

    for days in resBTC:
        if days['low']>0:
            row=BTC(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
            db.session.add(row)
            db.session.commit()

@event.listens_for(BTC_forecasts.__table__, 'after_create')
def create_BTC_forecasts(*args, **kwargs):

    btc=BTC.query.all()
    btc=pd.DataFrame(BTC.toDICT(btc))
    btc_preds = get_predictions(btc)
    
    for days in btc_preds:
        row=BTC_forecasts(close=days)
        db.session.add(row)
        db.session.commit()




class ETH(db.Model):
    __tablename__ = 'eth'
    time = db.Column(db.Float(), primary_key=True)
    high = db.Column(db.Float())
    low = db.Column(db.Float())
    open = db.Column(db.Float())
    close = db.Column(db.Float())
    volumeto = db.Column(db.Float())
    volumefrom = db.Column(db.Float())
    
    @classmethod
    def get_historical(self):
        return self.query.all()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

    def toJSON(self):
        return self.toDICT()
    
    def save(self):
        db.session.add(self)
        db.session.commit()


class ETH_forecasts(db.Model):

    __tablename__ = 'eth_predictions'
    close = db.Column(db.Float(), primary_key=True)

    @classmethod
    def get_historical(self):
        return self.query.all()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

    def toJSON(self):
        return self.toDICT()
    
    def save(self):
        db.session.add(self)
        db.session.commit()


   

@event.listens_for(ETH.__table__, 'after_create')
def create_ETH(*args, **kwargs):

    api_key='api_key={ea0232c4ea8a3007655f1518de6af8ea6c4a5e546ddf83988ec885db9600a11e}'
    ethUrl='https://min-api.cryptocompare.com/data/v2/histoday?fsym=ETH&tsym=USD&allData=true&'
    resETH = requests.get(ethUrl+api_key).json()['Data']['Data']

    for days in resETH:
        if days['low']>0:
            row=ETH(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
            db.session.add(row)
            db.session.commit()


@event.listens_for(ETH_forecasts.__table__, 'after_create')
def create_ETH_forecasts(*args, **kwargs):

    eth=ETH.query.all()
    eth=pd.DataFrame(ETH.toDICT(eth))

    eth_preds = get_predictions(eth)

    for days in eth_preds:
        row=ETH_forecasts(close=days)
        db.session.add(row)
        db.session.commit()


class XMR(db.Model):
    __tablename__ = 'xmr'
    time = db.Column(db.Float(), primary_key=True)
    high = db.Column(db.Float())
    low = db.Column(db.Float())
    open = db.Column(db.Float())
    close = db.Column(db.Float())
    volumeto = db.Column(db.Float())
    volumefrom = db.Column(db.Float())
    
    @classmethod
    def get_historical(self):
        return self.query.all()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

    def toJSON(self):
        return self.toDICT()
    
    def save(self):
        db.session.add(self)
        db.session.commit()


class XMR_forecasts(db.Model):

    __tablename__ = 'xmr_predictions'
    close = db.Column(db.Float(), primary_key=True)

    @classmethod
    def get_historical(self):
        return self.query.all()

    def toDICT(rset):
        result = defaultdict(list)
        for obj in rset:
            instance = inspect(obj)
            for key, x in instance.attrs.items():
                result[key].append(x.value)
        return result

    def toJSON(self):
        return self.toDICT()
    
    def save(self):
        db.session.add(self)
        db.session.commit()


@event.listens_for(XMR.__table__, 'after_create')
def create_XMR(*args, **kwargs):

    api_key='api_key={ea0232c4ea8a3007655f1518de6af8ea6c4a5e546ddf83988ec885db9600a11e}'
    xmrUrl='https://min-api.cryptocompare.com/data/v2/histoday?fsym=XMR&tsym=USD&allData=true&'
    resXMR = requests.get(xmrUrl+api_key).json()['Data']['Data']

    for days in resXMR:
        if days['low']>0:
            row=XMR(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
            db.session.add(row)
            db.session.commit()



@event.listens_for(XMR_forecasts.__table__, 'after_create')
def create_XMR_forecasts(*args, **kwargs):

    xmr=XMR.query.all()
    xmr=pd.DataFrame(XMR.toDICT(xmr))

    xmr_preds = get_predictions(xmr)

    for days in xmr_preds:
        row=XMR_forecasts(close=days)
        db.session.add(row)
        db.session.commit()

# <<<<<<< Updated upstream
# # @scheduler.task('interval', id='update_daily_values', seconds=5)
# # def daily_db_update():
# =======
@scheduler.task('interval', id='update_daily_values', hours=24)
def daily_db_update():
    with scheduler.app.app_context():
        
        api_key='api_key={ea0232c4ea8a3007655f1518de6af8ea6c4a5e546ddf83988ec885db9600a11e}'
        btcUrl_day='https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=1&'
        resBTC = requests.get(btcUrl_day+api_key).json()['Data']['Data']

        day_btc = resBTC[1]

        row1=BTC(time=day_btc['time'],high=day_btc['high'],low=day_btc['low'],open=day_btc['open'],close=day_btc['close'],volumeto=day_btc['volumeto'],volumefrom=day_btc['volumefrom'])
        db.session.add(row1)

        try:
            db.session.commit()
        except exc.SQLAlchemyError:
            pass

        ethUrl_day='https://min-api.cryptocompare.com/data/v2/histoday?fsym=ETH&tsym=USD&&limit=1&'
        resETH = requests.get(ethUrl_day+api_key).json()['Data']['Data']

        day_eth = resETH[1]
       
        row2=ETH(time=day_eth['time'],high=day_eth['high'],low=day_eth['low'],open=day_eth['open'],close=day_eth['close'],volumeto=day_eth['volumeto'],volumefrom=day_eth['volumefrom'])
        db.session.add(row2)

        try:
            db.session.commit()
        except exc.SQLAlchemyError:
            pass 
       

        xmrUrl_day='https://min-api.cryptocompare.com/data/v2/histoday?fsym=XMR&tsym=USD&&limit=1&'
        resXMR = requests.get(xmrUrl_day+api_key).json()['Data']['Data']

        day_xmr = resXMR[1]
        
        row3=XMR(time=day_xmr['time'],high=day_xmr['high'],low=day_xmr['low'],open=day_xmr['open'],close=day_xmr['close'],volumeto=day_xmr['volumeto'],volumefrom=day_xmr['volumefrom'])
        db.session.add(row3)
        try:
            db.session.commit()
        except exc.SQLAlchemyError:
            pass 


@scheduler.task('interval', id='make_weekly_predictions', weeks=1)
def daily_db_update():

    with scheduler.app.app_context():
        btc=BTC.query.all()
        btc=pd.DataFrame(BTC.toDICT(btc))
        btc_preds = get_predictions(btc)
        
        for days in btc_preds:
            row=BTC_forecasts(close=days)
            db.session.add(row)
            db.session.commit()
        
        eth=ETH.query.all()
        eth=pd.DataFrame(ETH.toDICT(eth))

        eth_preds = get_predictions(eth)

        for days in eth_preds:
            row=ETH_forecasts(close=days)
            db.session.add(row)
            db.session.commit()


        xmr=XMR.query.all()
        xmr=pd.DataFrame(XMR.toDICT(xmr))

        xmr_preds = get_predictions(xmr)

        for days in xmr_preds:
            row=XMR_forecasts(close=days)
            db.session.add(row)
            db.session.commit()

def toDICT(rset):
    result = defaultdict(list)
    for obj in rset:
        instance = inspect(obj)
        for key, x in instance.attrs.items():
            result[key].append(x.value)
    return result
    
def get_crypto():
    btc=BTC.query.all()
    eth=ETH.query.all()
    xmr=XMR.query.all()
    btc=pd.DataFrame(BTC.toDICT(btc))
    eth=pd.DataFrame(ETH.toDICT(eth))
    xmr=pd.DataFrame(XMR.toDICT(xmr))
    data=[btc,eth,xmr]

    return data
    
def get_eth_data():

    eth=ETH.query.all()
    eth_pred=ETH_forecasts.query.all()


    eth_df = pd.DataFrame(ETH.toDICT(eth))
    start = len(eth_df)-45

    eth_df = eth_df[['close']]
    eth_actual = (eth_df.iloc[start:,].values.ravel()).tolist()

    idx_actual = [i for i in range(0, len(eth_actual))]
    json_actuals = []

    for actual,idx in zip(eth_actual, idx_actual):

        json_actuals.append({'x': idx, 'y': actual })
    
    
    
    idx_pred = [i for i in range(len(eth_actual), len(eth_actual)+len(eth_pred))]
    json_preds = []
    json_preds.append(json_actuals[-1])
    

    for pred,idx in zip(eth_pred, idx_pred):

        json_preds.append({'x':idx ,'y':pred.close })

    
    return json.dumps(json_actuals),json.dumps(json_preds)

def get_btc_data():

    btc=BTC.query.all()
    btc_pred=BTC_forecasts.query.all()


    btc_df = pd.DataFrame(BTC.toDICT(btc))
    start = len(btc_df)-45

    btc_df = btc_df[['close']]
    btc_actual = (btc_df.iloc[start:,].values.ravel()).tolist()

    idx_actual = [i for i in range(0, len(btc_actual))]
    json_actuals = []

    for actual,idx in zip(btc_actual, idx_actual):

        json_actuals.append({'x': idx, 'y': actual })
    
    
    
    idx_pred = [i for i in range(len(btc_actual), len(btc_actual)+len(btc_pred))]
    json_preds = []
    json_preds.append(json_actuals[-1])
    

    for pred,idx in zip(btc_pred, idx_pred):

        json_preds.append({'x':idx ,'y':pred.close })

  
    return json.dumps(json_actuals),json.dumps(json_preds)

def get_xmr_data():

    xmr=XMR.query.all()
    xmr_pred=XMR_forecasts.query.all()


    xmr_df = pd.DataFrame(XMR.toDICT(xmr))
    start = len(xmr_df)-45

    xmr_df =  xmr_df[['close']]
    xmr_actual = (xmr_df.iloc[start:,].values.ravel()).tolist()

    idx_actual = [i for i in range(0, len(xmr_actual))]
    json_actuals = []

    for actual,idx in zip(xmr_actual, idx_actual):

        json_actuals.append({'x': idx, 'y': actual })
    
    
    
    idx_pred = [i for i in range(len(xmr_actual), len(xmr_actual)+len(xmr_pred))]
    json_preds = []
    json_preds.append(json_actuals[-1])
    

    for pred,idx in zip(xmr_pred, idx_pred):

        json_preds.append({'x':idx ,'y':pred.close })

    # print(json_preds)
    return json.dumps(json_actuals),json.dumps(json_preds)

def get_predictions(df):

    import numpy as np
    import ta
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error
    import tensorflow as tf
    

    data_df = df
    data_df = data_df.drop(columns=['time','volumefrom'])

    data_df = data_df.dropna()

    high = data_df.high
    low = data_df.low
    close = data_df.close
    volume = data_df.volumeto

    start = 0
    end = len(data_df)


    x_axis = [i for i in range(start,len(data_df))]


    RSI = ta.momentum.StochRSIIndicator(close)
    rsi_k = RSI.stochrsi_k()
    rsi_d = RSI.stochrsi_d()

    MACD_indicator = ta.trend.MACD(close)
    macd = MACD_indicator.macd_diff()

    KD = ta.momentum.StochasticOscillator(close, high, low)
    kd = KD.stoch()

    OBV = ta.volume.OnBalanceVolumeIndicator(close, volume)
    obv = OBV.on_balance_volume()

    atr = ta.volatility.average_true_range(high, low, close)

    data_df = data_df.assign(rsi_k=rsi_k)
    data_df = data_df.assign(rsi_d=rsi_d)
    data_df = data_df.assign(macd=macd)
    data_df = data_df.assign(kd=kd)
    data_df = data_df.assign(obv=obv)
    data_df = data_df.assign(atr=atr)

    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)

    data_df = data_df.drop(columns=['volumeto'])
    
    x = len(data_df)%8
    n = len(data_df)-x


    temporal = [[0,1,0,1,0,1,0,1] for i in range(0,n,8)]

    temporal = np.asarray(temporal).ravel().tolist()

    for i in range(0,x):
        
        if i < 4:
            temporal.append(0)
        else:
            temporal.append(1)
   

    data_df = data_df.assign(time=temporal)


    #-------------------------------------------------------------------------------------------------------------------------
    #_________________________________________________________________________________________________________________________
    #_________ SPLIT DATASET - TRAIN/TEST ____________________________________________________________________________________



    in_window = 70
    out_window = 14
    test_len = out_window


    num_features = len(data_df.columns)
    dataset_len = len(data_df)
    train_len = len(data_df)-test_len


    data = data_df.iloc[:,0:num_features].values
    close = data_df[['close']].values       


    #____________________________________________
    #---------- standard scaling ----------------
    sc = StandardScaler()
    sc2 = StandardScaler()


    train_data = sc.fit_transform(data[:train_len,:])
    train_close = sc2.fit_transform(np.asarray(close[:train_len]).reshape(-1,1))

    #_________________________________________________________________________________________
    #___________Train/Validation Data:________________________________________________________

    x_train = []
    y_train = []

    x_valid = []
    y_valid = []


    for i in range(in_window,len(train_data)-out_window-out_window+1):
        
        x_train.append(train_data[i-in_window:i,:])
        y_train.append(train_close[i:i+out_window,:])
        
        x_valid.append(train_data[i-in_window+out_window:i+out_window,:])
        y_valid.append(train_close[i+out_window:i+out_window+out_window,:])
        
        

    #______reshape train-data to lstm expected shape_______

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],num_features))

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)

    x_valid = np.reshape(x_valid,(x_valid.shape[0],x_valid.shape[1],num_features))

    #_________________________________________________________________________________________
    #______________Test Data:_________________________________________________________________


    #--------- standard-scaling ----------

    test_data = sc.transform(data[train_len-in_window:,:])
    test_close = sc2.transform(np.asarray(close[train_len-in_window:,:]).reshape(-1,1))

    x_test = []
    y_test = []


    for i in range(in_window,len(test_data)-out_window+1):
        
        x_test.append(test_data[i-in_window:i,:])
        y_test.append(test_close[i:i+out_window,:])


    #______reshape test-data to lstm expected shape_______

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    x_test = np.reshape(x_test,( x_test.shape[0], x_test.shape[1],num_features))

    #_________________________________________________________________________________________________________________________

    def build_model(in_window, out_window, num_features):
        
        inputs = tf.keras.layers.Input(shape=(in_window, num_features))
        
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(inputs)
        
        layer =  tf.keras.layers.Dense(32, kernel_initializer='lecun_normal', activation='selu')(layer)

        layer =  tf.keras.layers.Dropout(0.4)(layer)

        outputs = tf.keras.layers.Dense(out_window)(layer)
        
        model =tf.keras.models.Model(inputs, outputs)
        
   
        #opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        opt = 'sgd'
        
        #loss = tf.keras.losses.Huber() 
        loss = 'mean_squared_error'
        
        model.compile(optimizer=opt, loss=loss, metrics=['mape'])
        
        return model
        
        
        
        
    model_dnn = build_model(in_window, out_window, num_features)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)

    model_dnn.summary()
    hist_simple = model_dnn.fit(x_train, y_train, epochs=80, batch_size=8, callbacks=[callback], shuffle=False, validation_data=(x_valid, y_valid))


    y_pred = model_dnn.predict(x_test)


    #for y, pred in zip(y_test, y_pred):
    #    print("MSE: ", mean_squared_error(y, pred))
    #    print("MAPE: ", mean_absolute_percentage_error(y, pred))
    #    print('________________________________')
        
    y_pred = sc2.inverse_transform(y_pred)
    

    y_pred = y_pred[0]
    return y_pred
