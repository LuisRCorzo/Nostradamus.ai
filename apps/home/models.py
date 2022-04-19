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

@scheduler.task('interval', id='update_daily_values', seconds=5)
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
    start = len(eth_df)-100

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
    start = len(btc_df)-100

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

    print(json_preds)
    return json.dumps(json_actuals),json.dumps(json_preds)

def get_xmr_data():

    xmr=XMR.query.all()
    xmr_pred=XMR_forecasts.query.all()


    xmr_df = pd.DataFrame(XMR.toDICT(xmr))
    start = len(xmr_df)-100

    xmr_df = xmr_df[['close']]
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

    print(json_preds)
    return json.dumps(json_actuals),json.dumps(json_preds)

def get_predictions(df):

    import numpy as np
    #import matplotlib.pyplot as plt
    import ta
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error
    import tensorflow as tf
    

    data_df = df
    data_df = data_df.drop(columns=['time','volumefrom'])

    data_df = data_df.dropna()

    high = data_df.high
    low = data_df.low
    close = data_df.close

    start = 0
    end = len(data_df)


    x_axis = [i for i in range(start,len(data_df))]


    ichimoku = ta.trend.IchimokuIndicator(high, low)
    span_a = ichimoku.ichimoku_a()
    span_b = ichimoku.ichimoku_b()

    RSI = ta.momentum.RSIIndicator(close)
    rsi = RSI.rsi()

    MACD_indicator = ta.trend.MACD(close)
    MACD = MACD_indicator.macd_diff()

    bollinger_indicator = ta.volatility.BollingerBands(close)
    bollinger_high = bollinger_indicator.bollinger_hband()
    bollinger_low = bollinger_indicator.bollinger_lband()

    wma1 = ta.trend.wma_indicator(close,18)
    wma2 = ta.trend.wma_indicator(close,36)
    hull_inp = (2*(wma1))-wma2
    HULL = ta.trend.wma_indicator(hull_inp,6)

    x_axis = x_axis[start:end]
    close = close[start:end]

    span_a = span_a[start:end]
    span_b = span_b[start:end]

    rsi = rsi[start:end]
    MACD = MACD[start:end]

    data_df = data_df.assign(ichimoku_span_a=span_a)
    data_df = data_df.assign(ichimoku_span_b=span_b)

    data_df = data_df.assign(bollinger_high=bollinger_high)
    data_df = data_df.assign(bollinger_low=bollinger_low)

    data_df = data_df.assign(hull=HULL)

    data_df = data_df.assign(RSI=rsi)
    data_df = data_df.assign(MACD=MACD)

    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)

    #data_df = data_df.drop(columns=['ichimoku_span_a','ichimoku_span_b','bollinger_high','bollinger_low','RSI','MACD'])


    #-------------------------------------------------------------------------------------------------------------------------
    #_________________________________________________________________________________________________________________________
    #_________ SPLIT DATASET - TRAIN/TEST ____________________________________________________________________________________



    in_window = 35
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
        
        layer = tf.keras.layers.LSTM(in_window, return_sequences=False)(inputs)
        
        #layer = tf.keras.layers.LSTM(in_window)(layer)
        
        layer = tf.keras.layers.Dense(in_window)(layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)

        outputs = tf.keras.layers.Dense(out_window)(layer)
        
        model =tf.keras.models.Model(inputs, outputs)
        
        
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        #opt = 'Adam'
        #opt = 'sgd'
        
        #loss = tf.keras.losses.Huber() 
        #loss = 'mse'
        loss = 'msle'
        
        model.compile(optimizer=opt, loss=loss, metrics=['mape'])
        
        return model
        
        
        
        
    model_dnn = build_model(in_window, out_window, num_features)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model_dnn.summary()
    hist_simple = model_dnn.fit(x_train, y_train, epochs=100, batch_size=6, callbacks=[callback], shuffle=False, validation_data=(x_valid, y_valid))

    #plt.plot(hist_simple.history['loss'])
    #plt.plot(hist_simple.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'val'], loc='upper left')
    #plt.show()

    y_pred = model_dnn.predict(x_test)


    #for y, pred in zip(y_test, y_pred):
    #    print("MSE: ", mean_squared_error(y, pred))
    #    print("MAPE: ", mean_absolute_percentage_error(y, pred))
    #    print('________________________________')
        
    y_pred = sc2.inverse_transform(y_pred)
    

    '''
    for i in range(train_len,dataset_len-out_window):
        
        
        width = .89
        width2 = .12

        training_df = data_df.iloc[i-70:i,:]
        up = training_df[training_df.close>= training_df.open]
        down = training_df[training_df.close< training_df.open]

        
        actual_df = data_df.iloc[i:i+out_window,:]
        up_actual = actual_df[actual_df.close >= actual_df.open]
        down_actual = actual_df[actual_df.close < actual_df.open]
        
        
        pred_df = pd.DataFrame(y_pred[i-train_len,:])
        pred_df.columns=['Pred']
        pred_df = pred_df.set_index(actual_df.index)
        
        
        
        print("MAE: ", mean_absolute_error(actual_df.close, y_pred[i-train_len,:]))
        print("MAPE: ", mean_absolute_percentage_error(actual_df.close, y_pred[i-train_len,:]))
        
        
        #plotting predictions 
        ylim_high = actual_df.high.max()
        ylim_low = actual_df.low.min()
        
        ylim_high2 = training_df.high.max()
        ylim_low2 = training_df.low.min()
        
        if ylim_high < ylim_high2:
            ylim_high = ylim_high2
        
        if ylim_low > ylim_low2:
            ylim_low = ylim_low2
            
            
        #define colors to use
        col1 = 'green'
        col2 = 'red'
    
        plt.figure(figsize=(15, 9))
        
        #plot up prices
        plt.bar(up.index, up.close-up.open, width, bottom=up.open, color=col1,)
        plt.bar(up.index, up.high-up.close, width2, bottom=up.close, color=col1)
        plt.bar(up.index, up.low-up.open, width2, bottom=up.open, color=col1)

        #plot down prices
        plt.bar(down.index, down.close-down.open, width, bottom=down.open, color=col2)
        plt.bar(down.index, down.high-down.open, width2, bottom=down.open, color=col2)
        plt.bar(down.index, down.low-down.close, width2, bottom=down.close, color=col2)
        
        plt.bar(up_actual.index, up_actual.close-up_actual.open, width, bottom=up_actual.open, color=col1, alpha=0.6)
        plt.bar(up_actual.index, up_actual.high-up_actual.close, width2, bottom=up_actual.close, color=col1, alpha=0.6)
        plt.bar(up_actual.index, up_actual.low-up_actual.open, width2, bottom=up_actual.open, color=col1, alpha=0.6)

        plt.bar(down_actual.index, down_actual.close-down_actual.open,width, bottom=down_actual.open, color=col2, alpha=0.6)
        plt.bar(down_actual.index, down_actual.high-down_actual.open, width2, bottom=down_actual.open, color=col2, alpha=0.6)
        plt.bar(down_actual.index, down_actual.low-down_actual.close, width2, bottom=down_actual.close, color=col2, alpha=0.6)
        
        
        pred_to_plot = np.insert(pred_df['Pred'].values,0,data_df.iloc[i,3])
        
        plot_start = data_df.loc[i].name
        
        plot_idx = [i for i in range(plot_start, plot_start+out_window+1)]
        
        plt.plot(plot_idx,pred_to_plot,color='purple')
        
        #plt.ylim(ylim_low-10, ylim_high+10)
        
        #rotate x-axis tick labels
        plt.xticks(rotation=45, ha='right')

        #display candlestick chart
        plt.show()
    '''
    y_pred = y_pred[0]
    return y_pred
