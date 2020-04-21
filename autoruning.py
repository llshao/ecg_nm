from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import json
import keras
import numpy as np
import os
import random
import time

import network
import load
import util
import train
import robust_autoencoder
class Argparse():
    def __init__(self):
        self.config_file = []
        self.experiment = []
def run_autoencoder(params):
    encoder = robust_autoencoder.RobustAutoencoder()
    encoder.train_scaled = encoder.load_dataset(params['data_json'])#train[0]: ecg_data train[1]: labels 
    encoder.dev_scaled = encoder.load_dataset(params['dev_json'])
    
    # scale the data
    encoder.train_scaled, _, _ = encoder.scale_input(encoder.train_scaled[0])
    encoder.dev_scaled, _, _ = encoder.scale_input(encoder.dev_scaled[0])
    
    # add noisy
    encoder.train_noisy = encoder.add_noise_snr(encoder.train_scaled, des_snr=params['des_snr'], noise_type=params['noisy_type'])
    encoder.dev_noisy = encoder.add_noise_snr(encoder.dev_scaled, des_snr=params['des_snr'], noise_type=params['noisy_type'])
    
    # build autoencoder
    encoder.build_encoder(params)
    
    # train the autoencoder
    encoder.train_encoder(params)
    
    # reconstruct the noisy signals
    encoder.reconstruct(params)
    
    # store the reconstructed/noisy signals
    encoder.store_results(params)
    
    reset_keras(encoder.model)
    del encoder.history
    gc.collect()
    
    
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import gc
# Reset Keras Session
def reset_keras(model=None):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))
      
if __name__ == '__main__':
    AUTO_ENCODER = True
    # initial parmas
    tran_json_prefix = ['_train_recon','_train']
    dev_json_prefix = ['_dev_recon','_dev']
    params = {
        'data_json': "examples/cinc17/train.json",
        'dev_json': "examples/cinc17/dev.json",
        'des_snr' : 20,
        'noisy_type': 4, #0:gaussian 1: cauchy 4:mix of 1,2
        'train_type': 1, #0: clean-clean 1: noisy-clean 2: noisy-noisy
        'loss': 'mcc',
        'MAX_EPOCHS': 80,
        'batch_size': 256,
        'input_size': 256,
        'hidden_size': 64,
        'lr': 0.01,
        'lambda_w': 4e-5
             }
    db_list =  [5, 10, 15, 20, 25, 30, 35, 40]
    loss_list = ['mcc','mse']
    noisy_type = [4, 0, 1]

    for noise in noisy_type:
            for loss in loss_list:
                train_jsons = []
                dev_jsons = []
                experiments = []
                for db in db_list:
                    params['noisy_type'] = noise
                    params['des_snr'] = db
                    params['loss'] = loss
                    # update file path based on params
                    params['store_data_folder'] = 'examples/cinc17/mcc_transformed/'
                    params['experiment_name'] = 'noisy'+str(params['noisy_type'])+'_db'+str(params['des_snr'])+'_'+params['loss']
                    params['model_name'] = 'autoencoder_model/'+'train'+ str(params['train_type'])+'_'+params['experiment_name']+ '_autoencoder.h5'
                    params['store_json'] = 'examples/cinc17/'+params['experiment_name']
                    for t,d in zip(tran_json_prefix, dev_json_prefix):
                        train_jsons.append(params['store_json'] + t +'.json')
                        dev_jsons.append(params['store_json'] + d +'.json')
                        experiments.append(params['experiment_name']+t+d)

                    # run auto_encoder
                    if AUTO_ENCODER == True:
                        reset_keras()
                        run_autoencoder(params)

                ##########################################################
                for t, d, e in zip(train_jsons, dev_jsons, experiments):
                    args = Argparse()
                    args.config_file = 'examples/cinc17/config.json'
                    args.experiment = e
                    params_cnn = json.load(open(args.config_file, 'r'))
                    params_cnn['train'] = t
                    params_cnn['dev'] = d

                    reset_keras()
                    train.train(args, params_cnn)