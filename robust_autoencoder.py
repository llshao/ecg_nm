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

import copy as cp
# params for ecg_seg & save file path
STEP = 256
POST_FIX_INDEX = -10 # file_path[-10:] 'AXXXX.mat'

class RobustAutoencoder():
    def __init__(self):
        # data storage
        self.train = None
        self.dev = None
        
        self.train_scaled = None
        self.dev_scaled = None
        
        self.train_noisy = None
        self.dev_noisy = None
        
        self.train_recon = None
        self.dev_recon = None
        
        self.train_mcc = None
        self.dev_mcc = None
        
        # model storage
        self.model = None
        self.history = None
        
    
    def cal_pixel_mse(self, origin, rec):
        '''
        calculate the mse errors for origin and reconstructed data
        '''
        origin = np.array(origin)
        rec = np.array(rec)
        sample, _ = origin.shape
        mse = []
        for i in range(sample):
            mse.append(np.mean((origin[i]-rec[i])**2))
        return np.array(mse), np.mean(mse)
    
    def load_dataset_path(self, data_json):
        '''
        given json file to return the ecgs, lables and orignal path
        ----
        params
        data_json: str. path of the json file
        -----
        return

        '''
        import tqdm
        with open(data_json, 'r') as fid:
            data = [json.loads(l) for l in fid]# just load the filename in data
        labels = []; ecgs = []; paths = []
        for d in tqdm.tqdm(data):
            paths.append(d['ecg'])
            labels.append(d['labels'])
            ecgs.append(self.load_ecg_path(d['ecg']))## load .mat file
        return ecgs, labels, paths
    
    def load_ecg_path(self, record):
        '''
        load .mat files and reshape for given STEP=256
        '''
        import scipy.io as sio
        if os.path.splitext(record)[1] == ".npy":
            ecg = np.load(record)
        elif os.path.splitext(record)[1] == ".mat":
            ecg = sio.loadmat(record[-35:])['val'].squeeze()# hack for different path
        else: # Assumes binary 16 bit integers
            with open(record, 'r') as fid:
                ecg = np.fromfile(fid, dtype=np.int16)

        trunc_samp = STEP * int(len(ecg) / STEP)
        
        return ecg[:trunc_samp]
    
    def load_dataset(self, data_json):
        '''
        return segmented ecgs and lables
        '''
        import tqdm
        with open(data_json, 'r') as fid:
            data = [json.loads(l) for l in fid]# just load the filename in data
        labels = []; ecgs = []
        for d in tqdm.tqdm(data):
            labels.extend(d['labels'])
            ecgs.extend(self.load_ecg(d['ecg']))## load .mat file
        return ecgs, labels

    def load_ecg(self, record):
        '''
        load .mat files and reshape for given STEP=256
        '''
        import scipy.io as sio
        if os.path.splitext(record)[1] == ".npy":
            ecg = np.load(record)
        elif os.path.splitext(record)[1] == ".mat":
            ecg = sio.loadmat(record[-35:])['val'].squeeze()# hack for different path
        else: # Assumes binary 16 bit integers
            with open(record, 'r') as fid:
                ecg = np.fromfile(fid, dtype=np.int16)

        trunc_samp = STEP * int(len(ecg) / STEP)

        return ecg[:trunc_samp].reshape(-1,STEP)
    
    def scale_input(self, data):
        '''
        scale the data to [0, 1]
        '''
        data_scaled = [(d - np.min(d))/(np.max(d)-np.min(d)) for d in data]
        scales = [(np.max(d)-np.min(d)) for d in data]
        bias = [np.min(d) for d in data]
        return np.array(data_scaled), np.array(scales), np.array(bias)
    
    def scale_back(self, data, scales, bias):
        origin_scale = []
        for d, s, b in zip(data, scales, bias):
            origin_scale.append(d*s + b)
        return np.array(origin_scale)
    
    def transform(self, origin, model):
        '''
        transform the orignal ecg sginal using the given model
        ----
        params
        origin: list(list)  ecg signals. 
        model: trained  autoencoder
        -----
        return
        tansformed: ndarray of transformed data
        '''
        transformed = []
        for data in origin:
            # reshape and scale
            data = data.reshape(-1,STEP)
            data_scaled, scales, bias = self.scale_input(data)
            # model predict and scale back
            origin_scale = scale_back(model.predict(data_scaled), scales, bias)
            transformed.append(origin_scale.reshape(-1,1).squeeze())
        return np.array(transformed).squeeze() # keep same dimension as origin[0] data


    def add_noise_snr(self, signals, des_snr, noise_type = 4):
        '''
        add noise to the signal for a given SNR (des_snr)
        ref: https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
        ------
        params
        signals: ndarray  original signals
        signal_type: int  0: gaussain; 1: cauchy; 2: possion ; 
                          3: speckle: Multiplicative noise using out = image + n*image 
                          4: mix of 0-1 (default)
        des_snr: int desired SNR
        -----
        return
        noisy_signals: ndarray 
        '''
        noisy_signals = []
        for sig in signals:
            # get the power of signal and desired power of noise
            sig_power = np.sum(np.array(sig)**2)/len(sig)
            des_noise_power = sig_power/np.math.pow(10,des_snr/10)
            # generate noise and scale accordingly based on desired SNR
            noise = self.generate_signals(np.array(sig), noise_type)
            noise_power = np.sum(np.array(noise)**2)/len(noise)
            noise = np.sqrt(des_noise_power/noise_power)*noise
            # append
            noisy_signals.append(sig + noise)
        return np.array(noisy_signals)

    def generate_signals(self, sig, noise_type = 4):
        '''
        generate noises for given sig and noise type

        '''
        # TODO: add different noises
        if noise_type == 0: #gaussian
            noise = np.random.normal(size=sig.shape)
        elif noise_type == 1: # cauchy
            from scipy.stats import cauchy
            noise = cauchy.rvs(size=sig.shape) 
        elif noise_type == 2: # poisson
            noise = np.random.poisson(size=sig.shape)
        elif noise_type == 3: # speckle: Multiplicative noise using out = image + n*image,where  n is gaussion noise with specified mean & variance.
            noise = np.random.randn(size=sig.shape)*sig
        else:# mix of 0 & 1 
            from scipy.stats import cauchy
            noise = np.random.normal(size=sig.shape) + cauchy.rvs(size=sig.shape) 
        return noise

    
    def mcc_loss(self, y_actual, y_predicted, sigma = 0.2):
        import keras.backend as K
        diff = y_actual-y_predicted
        mcc_loss = -1/K.sqrt(2*K.variable(np.pi))/sigma*K.exp(-diff**2/2/K.variable(sigma)**2)
        return K.mean(mcc_loss)
    
    def build_encoder(self, params):
        input_size = params['input_size']
        hidden_size = params['hidden_size']
        lr = params['lr']
        lambda_w = params['lambda_w']
        
        
        from keras.models import Model
        from keras.layers import Input
        from keras.layers.core import Dense
        from keras.regularizers import l2
  
        x = Input(shape=(input_size,))
        hidden_1 = Dense(hidden_size, activation='relu', kernel_regularizer=l2(lambda_w), bias_regularizer=l2(lambda_w))(x)
        #h = Dense(code_size, activation='relu')(hidden_1)
        hidden_2 = Dense(hidden_size, activation='relu', kernel_regularizer=l2(lambda_w), bias_regularizer=l2(lambda_w))(hidden_1)
        r = Dense(input_size, activation='sigmoid',kernel_regularizer=l2(lambda_w), bias_regularizer=l2(lambda_w))(hidden_2)

        self.model = Model(inputs=x, outputs=r)
        from keras.optimizers import Adam
        optimizer = Adam(
            lr = lr)
        if params['loss'] == 'mcc':
            self.model.compile(optimizer='adam', loss = self.mcc_loss)  
        else:
            self.model.compile(optimizer='adam', loss = 'mse')  
        
    def train_encoder(self, params):
        # pass params 
        MAX_EPOCHS = params['MAX_EPOCHS']
        batch_size = params['batch_size']
        mcc_model_name = params['model_name']
        
        # callbacks
        from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta= 0.00005, verbose=1, patience=int(0.15*MAX_EPOCHS))
        checkpointer = ModelCheckpoint(filepath=mcc_model_name, monitor='val_loss', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                                      verbose=0, mode='auto', cooldown=0, min_lr=1e-6)
        # fit
        if params['train_type'] == 0:
            self.history = self.model.fit(self.train_scaled, self.train_scaled, batch_size=batch_size, epochs=MAX_EPOCHS, 
                                  shuffle = True, verbose=1, validation_data=(self.dev_scaled, self.dev_scaled), 
                                  callbacks = [checkpointer, reduce_lr, stopping])
        
        if params['train_type'] == 1:# train with noisy-clean pair
            self.history = self.model.fit(self.train_noisy, self.train_scaled, batch_size=batch_size, epochs=MAX_EPOCHS, 
                                  shuffle = True, verbose=1, validation_data=(self.dev_noisy, self.dev_scaled), 
                                  callbacks = [checkpointer, reduce_lr, stopping])
        
        if params['train_type'] == 2:
            self.history = self.model.fit(self.train_noisy, self.train_noisy, batch_size=batch_size, epochs=MAX_EPOCHS, 
                                  shuffle = True, verbose=1, validation_data=(self.dev_noisy, self.dev_noisy), 
                                  callbacks = [checkpointer, reduce_lr, stopping])
        
    def reconstruct(self, params):
        from keras.models import load_model
        if params['loss'] == 'mcc':
            saved_model = load_model(params['model_name'], custom_objects={'mcc_loss': self.mcc_loss})
        else:
            saved_model = load_model(params['model_name'])
        
        if params['train_type'] == 0: 
            self.train_recon = saved_model.predict(self.train_noisy)
            self.dev_recon = saved_model.predict(self.dev_noisy)
            
        if params['train_type'] == 1 :
            self.train_recon = saved_model.predict(self.train_noisy)
            self.dev_recon = saved_model.predict(self.dev_noisy)
         
        if params['train_type'] == 2 :
            self.train_recon = saved_model.predict(self.train_noisy)
            self.dev_recon = saved_model.predict(self.dev_noisy)
            
    def transform_segmented(self, origin, segments, model):
        '''
        transform the segmented ecg sginal using the given model
        ----
        params
        origin: list(list)  ecg signals. 
        segments: list(list) segmented ecg signal based on STEP = 256
        model: trained  autoencoder
        -----
        return
        tansformed: ndarray of transformed data
        '''
        transformed = []
        start = 0
        for data in origin:
            # reshape and scale
            data_cp = cp.copy(data)
            data_cp = data_cp.reshape(-1,STEP)
            _, scales, bias = self.scale_input(data_cp)
            # model predict and scale back
            data_scaled = segments[start:start+len(data_cp)]
            start += len(data_cp)
            origin_scale = self.scale_back(model.predict(data_scaled), scales, bias)
            transformed.append(origin_scale.reshape(-1,1).squeeze())
        return np.array(transformed).squeeze() # keep same dimension as origin[0] data  

    def scale_back_noisy(self, origin, segments):
        '''
        transform the segmented ecg sginal using the given model
        ----
        params
        origin: list(list)  ecg signals. 
        segments: list(list) segmented ecg signal based on STEP = 256
        -----
        return
        tansformed: ndarray of transformed data
        '''
        transformed = []
        start = 0
        for data in origin:
            # reshape and scale
            data_cp = cp.copy(data)
            data_cp = data_cp.reshape(-1,STEP)
            _, scales, bias = self.scale_input(data_cp)
            # model predict and scale back
            data_scaled = segments[start:start+len(data_cp)]
            start += len(data_cp)
            origin_scale = self.scale_back(data_scaled, scales, bias)
            transformed.append(origin_scale.reshape(-1,1).squeeze())
        return np.array(transformed).squeeze() # keep same dimension as origin[0] data
    
    def transform(self, origin, model):
        '''
        transform the orignal ecg sginal using the given model
        ----
        params
        origin: list(list)  ecg signals. 
        model: trained  autoencoder
        -----
        return
        tansformed: ndarray of transformed data
        '''
        transformed = []
        origin = np.array(origin)
        for data in origin:
            # reshape and scale
            data_cp = cp.copy(data)
            data_cp = data_cp.reshape(-1,STEP)
            data_scaled, scales, bias = self.scale_input(data_cp)
            # model predict and scale back
            origin_scale = self.scale_back(model.predict(data_scaled), scales, bias)
            transformed.append(origin_scale.reshape(-1,1).squeeze())
        return np.array(transformed).squeeze() # keep same dimension as origin[0] data
    

    def save_transformed(self, path, data, origin_path, experiment_name= 'e'):
        '''
        save the transformed data to the given path
        --------
        params:
        path: str  the path to store the transformed files
        data: ndarray transformed data
        origin_path: original path and name
        experiemnt_name: str identifier for certer experiments 
        '''
        import scipy.io as sio
        save_dir = self.make_save_dir(path, experiment_name)
        for d, origin in zip(data, origin_path):
            file_name = self.get_filename_for_saving(save_dir, origin)
            sio.savemat(file_name, {'val': d})
        return

    def make_save_dir(self, dirname, experiment_name):

        save_dir = os.path.join(dirname, experiment_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir

    def get_filename_for_saving(self, save_dir, origin):
        return os.path.join(save_dir, origin[POST_FIX_INDEX:])
    
    def save_json(self, json_path, mat_file_path, file_names, labels, experiment_name = 'e'):
        '''
        params:
        ---------------
        json_path: str   the path and name to store the json file
        mat_file_path: str   the path of previous stored .mat file
        file_names: list(str) the names of original .mat file
        labels: list(list(str)) the label for each seg (STEP=256) of ecg signals

        returns:
        --------------
        None
        '''
        with open(json_path, 'w') as fid:
            for name, label in zip(file_names, labels):
                save_dir = os.path.join(mat_file_path, experiment_name)
                datum = { 'ecg' : save_dir+'/'+name[POST_FIX_INDEX:],
                         'labels': label}
                json.dump(datum, fid)
                fid.write('\n')

    def store_results(self, params):
        '''
        store noisy data; 
        nosiy data mse reconstructed ; 
        noisy data mcc reconstructed;
        origin data mcc reconstructed;
        '''
        from keras.models import load_model
        if params['loss'] == 'mcc':
            saved_model = load_model(params['model_name'], custom_objects={'mcc_loss': self.mcc_loss})
        else:
            saved_model = load_model(params['model_name'])
            
        self.train = self.load_dataset_path(params['data_json'])#train[0]: ecg_data train[1]: labels train[2]:origin path
        self.dev = self.load_dataset_path(params['dev_json'])
        
        ## to be saved
        self.train_noisy = self.scale_back_noisy(self.train[0], self.train_noisy)
        self.dev_noisy = self.scale_back_noisy(self.dev[0], self.dev_noisy)
        #save_transformed(path_noisy_train, train_mix_db30, train_path[2])
        #save_json(train_json, 'examples/'+ path_noisy_train, train_path[2], train_path[1])
        self.save_transformed(params['store_data_folder'], self.train_noisy, 
                              self.train[2], 'train_'+ params['experiment_name'])
        self.save_json(params['store_json']+'_train'+'.json', params['store_data_folder'], 
                       self.train[2], self.train[1], 'train_'+params['experiment_name'])
        
        self.save_transformed(params['store_data_folder'], self.dev_noisy, self.dev[2], 
                              'dev_'+ params['experiment_name'])
        self.save_json(params['store_json']+'_dev'+'.json', params['store_data_folder'], 
                       self.dev[2], self.dev[1], 'dev_'+params['experiment_name'])  
        
        self.train_recon = self.scale_back_noisy(self.train[0], self.train_recon)
        self.dev_recon = self.scale_back_noisy(self.dev[0], self.dev_recon)
        #save_transformed(path_noisy_train_mcc, train_mix_db30_mcc_transformed, train_path[2])
        #save_json(mse_train_json, 'examples/'+ path_noisy_train_mse, train_path[2], train_path[1])
        self.save_transformed(params['store_data_folder'], self.train_recon, 
                              self.train[2], 'train_recon_'+ params['experiment_name'])
        self.save_json(params['store_json']+'_train_recon'+'.json', params['store_data_folder'], 
                       self.train[2], self.train[1], 'train_recon_'+params['experiment_name'])
        
        self.save_transformed(params['store_data_folder'], self.dev_recon, self.dev[2], 
                              'dev_recon_'+ params['experiment_name'])
        self.save_json(params['store_json']+'_dev_recon'+'.json', params['store_data_folder'], 
                       self.dev[2], self.dev[1], 'dev_recon_'+params['experiment_name'])  
        
        self.train_trans = self.transform(self.train[0], saved_model)
        self.dev_trans = self.transform(self.dev[0], saved_model)
        #save_transformed(path_origin_train_mcc, train_origin_mcc_transformed, train_path[2])
        #save_json(mcc_train_json, 'examples/'+ path_origin_train_mcc, train_path[2], train_path[1])
        self.save_transformed(params['store_data_folder'], self.train_trans, 
                              self.train[2], 'train_trans_'+ params['experiment_name'])
        self.save_json(params['store_json']+'_train_trans'+'.json', params['store_data_folder'], 
                       self.train[2], self.train[1], 'train_trans_'+params['experiment_name'])
        
        self.save_transformed(params['store_data_folder'], self.dev_trans, self.dev[2], 
                              'dev_trans_'+ params['experiment_name'])
        self.save_json(params['store_json']+'_dev_trans'+'.json', params['store_data_folder'], 
                       self.dev[2], self.dev[1], 'dev_trans_'+params['experiment_name'])  
         
            
if __name__ == "__main__":
    # initial parmas
    train_jsons = []
    dev_jsons = []
    tran_json_prefix = ['_train_','_train_recon']
    dev_json_prefix = ['_dev_','_dev_recon']
    params = {
        'data_json': "examples/cinc17/train.json",
        'dev_json': "examples/cinc17/dev.json",
        'des_snr' : 20,
        'noisy_type': 4,
        'train_type': 1, #0: clean-clean 1: noisy-clean 2: noisy-noisy
        'loss': 'mcc',
        'MAX_EPOCHS': 80,
        'batch_size': 128,
        'input_size': 256,
        'hidden_size': 64,
        'lr': 0.01,
        'lambda_w': 4e-5
             }
    # update file path based on params
    params['store_data_folder'] = 'exmaples/cinc17/mcc_transformed/'
    params['experiment_name'] = 'noisy'+str(params['noisy_type'])+'_db'+str(params['des_snr'])+'_'+params['loss']
    params['model_name'] = 'autoencoder_model/'+'train'+ str(params['train_type'])+'_'+params['experiment_name']+ '_autoencoder.h5'
    params['store_json'] = 'examples/cinc17/'+params['experiment_name']
    for t,d in zip(tran_json_prefix, dev_json_prefix):
        train_jsons.append(params['store_json'] + t +'.json')
        dev_jsons.append(params['store_json'] + t +'.json')
    # load origin data
    encoder = RobustAutoencoder()
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
    
    
    