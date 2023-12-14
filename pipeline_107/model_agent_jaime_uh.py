#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import logging
import time
import re
import logging
import datetime
import pandas as pd
from enum import Enum 
from math import fabs, floor
import json
import traceback

import paho.mqtt.client as mqtt
from urllib.parse import urlparse

from dotenv import load_dotenv
import numpy as np
import tensorflow as tf

### add additional libs or classes
from TiED_layers import TiED
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


# In[2]:


### modification
def prepare_input(data, input_len, pred_interval, coef, up_to_dn = True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        input_len: the length of a single sample
        pred_interval: the interval between the input and the target.
        up_to_dn: Sampling frequency.

    Returns:
        NumPy array of series framed for supervised learning.
    """

    df = data/coef
    sample_len = input_len + pred_interval # 1000+100 = 1100
    sample_num = int(df.shape[0]-sample_len+1)
    features = np.empty((sample_num,sample_len-pred_interval))
    targets = np.empty((sample_num,))
    
    if up_to_dn == True:
        flag = 1 
    else:
        flag = 0

    for j in range(0, sample_num):
        clip_start = j
        clip_end = clip_start+sample_len
        features[j,:] = df[clip_start:int(clip_end-pred_interval), flag].T#use upstream data
        targets[j] = df[clip_end-1, flag]
      
    return features, targets


# In[ ]:


def callback(body):
    t0 = datetime.datetime.now().replace(microsecond=0)

    # convert inbound data to data frames
    raw_data = json.loads(body)
    df_up_raw = pd.DataFrame(raw_data['up'])
    df_dn_raw = pd.DataFrame(raw_data['dn'])
  
    # rearrange data to be in a single data frame. 0:up, 1:dn
    row = min(df_up_raw.shape[0], df_dn_raw.shape[0])
    df = np.concatenate((df_up_raw.iloc[:row,:].psi.values.reshape(-1,1),
                df_dn_raw.iloc[:row,:].psi.values.reshape(-1,1)), 
                axis = 1) #(n,2)
    
    ### determine flow from upstream to downstream or vice versa
    if sum(df[:,0]>df[:,1])>=0.8*row:
        # upstream to downstream
        up_to_dn = True
        # get timestamp of last data point
        last_predition_ts = df_dn_raw.iloc[df_dn_raw.shape[0]-1, 0]
        Pred_Model = model_dn
    else:
        # downstream to upstream
        up_to_dn = False
        last_predition_ts = df_up_raw.iloc[df_up_raw.shape[0]-1, 0]
        Pred_Model = model_up
        
    # time shift and expected sampling freq
    time_shift = 0
    freq = 100

    ### run prediction
    model_input, pressure_tars = prepare_input(df, input_len = config_dict['input_len'], 
                                               pred_interval = config_dict['pred_interval'],
                                               coef = config_dict['coef'],
                                               up_to_dn = up_to_dn)
    
    pred_pressure_tars = Pred_Model.predict(model_input)

    ### look though range looking for largest change on predictions on each second
    flat_prediction = pred_pressure_tars.flatten() # predicted pressure values
    predition_diff = pressure_tars.flatten() - flat_prediction # difference between predicted and actual pressure values 
    predition_diff_sec = {}
    pred_sec = {}
    actual_sec = {}
    maxdiff = 0
    first_prediction_ts = last_predition_ts - len(predition_diff) / 100
    for tidx in range(0, len(predition_diff)): # save actual pressure and predicted value with interval 1 second
        sec = floor(first_prediction_ts + tidx / 100)
        if (sec not in predition_diff_sec or abs(predition_diff[tidx]) > maxdiff):
            maxdiff = abs(predition_diff[tidx])
            predition_diff_sec[sec] = predition_diff[tidx]
            pred_sec[sec] = pred_pressure_tars[tidx]
            actual_sec[sec] = pressure_tars[tidx]

    ### Aggregate the data
    per_sec_data = {}
    for ts, diff in predition_diff_sec.items():
        actual = actual_sec[ts]
        pred = pred_sec[ts][0]
        #per_sec_data[ts] = {'actual': 1.0 * actual, 'pred': 1.0 * pred, 'diff': 1.0 * diff}
        per_sec_data[ts] = {'actual': 1.0 * actual, 'pred': 1.0 * pred}

    ### send message out with prediction
    out_data = {'data': per_sec_data, 'analyst_id': raw_data['analyst_id'], 'flow direction': up_to_dn}
    json_data = json.dumps(out_data)

    t1 = datetime.datetime.now().replace(microsecond=0)
    logger.info("callback: took " + str(t1 - t0))

    return json_data


# In[ ]:


if __name__ == "__main__":
    try:
        load_dotenv()
        
        # set up logger
        logger = logging.getLogger('model-agent')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Read config data
        with open(os.getenv("CONFIG_JSON"), 'r') as f:
            config_dict = json.load(f)

        analyst_id = str(os.getenv("ANALYST_ID"))
        logger.info("Model for analyst " + analyst_id)

        ### open upstream model and downstream model
        #### upstream model
        weight_path_up = config_dict["weight_path_up"]        
        med_size_up = config_dict["med_size_up"] #1024
        outputsize_up = config_dict["outputsize_up"] #1

        m_in_up = Input([config_dict["input_len"],])
        m_out_up = TiED(en_lyers = 3, de_lyers = 3, med_size = med_size_up, outputsize = outputsize_up)(m_in_up)
        model_up= Model(m_in_up,m_out_up)  
        model_up.compile(loss = 'mse', metrics=['mae'])
        model_up.summary()
        model_up.load_weights(weight_path_up)
        
        #### downstream model
        weight_path_dn = config_dict["weight_path_dn"]        
        med_size_dn = config_dict["med_size_dn"] #1024
        outputsize_dn = config_dict["outputsize_dn"] #1

        m_in_dn = Input([config_dict["input_len"],])
        m_out_dn = TiED(en_lyers = 3, de_lyers = 3, med_size = med_size_dn, outputsize = outputsize_dn)(m_in_dn)
        model_dn= Model(m_in_dn,m_out_dn)  
        model_dn.compile(loss = 'mse', metrics=['mae'])
        model_dn.summary()
        model_dn.load_weights(weight_path_dn)
        
        # connect activities
        def on_connect(client, userdata, flags, rc):
            try:
                if (rc != 0):
                   logger.error('on_connect: Unable to connect to mqtt, rc=' + str(rc))
                   quit()
                logger.info("on_connect: success")
                client.subscribe("model-input/" + analyst_id)
            except:
               traceback.print_exc()

        def on_message(client, userdata, msg):
            try:
               logger.info("on_message: " + msg.topic + " mid: " + str(msg.mid))
               result = callback(msg.payload)
               client.publish("model-output/" + analyst_id, result)
            except:
               traceback.print_exc()
            
        def on_publish(client, userdata, mid):
            logger.info("on_publish: " + mid)

		# Spin up mqtt connection
        mqtt_client = mqtt.Client("model-agent-" + analyst_id)
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        mqtt_client.on_publish = on_publish
        
        o = urlparse(os.getenv("MQTT_URI"))
        mqtt_client.username_pw_set(o.username, password=o.password)
        mqtt_client.tls_set(ca_certs=os.getenv("CA_CERTS"))

        mqtt_client.connect(o.hostname, o.port, 60)

        ## consume data
        logger.info('waiting...')
        mqtt_client.loop_forever()

    except:
       logger.critical(sys.exc_info()[0])
       raise


# In[ ]:




