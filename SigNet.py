# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:24:27 2017

@author: sounak_dey and anjan_dutta
"""

import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import argparse
from webapp import app

#from keras.utils.visualize_util import plot

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras import backend as K
from SignatureDataGenerator import SignatureDataGenerator
import getpass as gp
import sys
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random
random.seed(1337)


# Create a session for running Ops on the Graph.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def euclidean_distance(vects):
    x, y = vects
    print (x)
    print (y)
    #return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True)))
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    
def create_base_network_signet(input_shape):
    
    seq = Sequential()
    seq.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1', subsample=(4, 4), input_shape= input_shape, 
                        init='glorot_uniform', dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))    
    seq.add(ZeroPadding2D((2, 2), dim_ordering='tf'))
    
    seq.add(Convolution2D(256, 5, 5, activation='relu', name='conv2_1', subsample=(1, 1), init='glorot_uniform',  dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    
    seq.add(Convolution2D(384, 3, 3, activation='relu', name='conv3_1', subsample=(1, 1), init='glorot_uniform',  dim_ordering='tf'))
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))
    
    seq.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', subsample=(1, 1), init='glorot_uniform', dim_ordering='tf'))    
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
#    model.add(SpatialPyramidPooling([1, 2, 4]))
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))
    seq.add(Dropout(0.5))
    
    seq.add(Dense(128, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform')) # softmax changed to relu
    
    return seq

def compute_accuracy_roc(predictions, labels):
   '''Compute ROC accuracy with a range of thresholds on distances.
   '''
   dmax = np.max(predictions)
   dmin = np.min(predictions)
   nsame = np.sum(labels == 1)
   ndiff = np.sum(labels == 0)
   
   step = 0.01
   max_acc = 0
   
   for d in np.arange(dmin, dmax+step, step):
       idx1 = predictions.ravel() <= d
       idx2 = predictions.ravel() > d
       
       tpr = float(np.sum(labels[idx1] == 1)) / nsame       
       tnr = float(np.sum(labels[idx2] == 0)) / ndiff
       acc = 0.5 * (tpr + tnr)       
#       print ('ROC', acc, tpr, tnr)
       
       if (acc > max_acc):
           max_acc = acc
           
   return max_acc
    
def read_signature_data(dataset, ntuples, height = 30, width = 100):
    
    images = []
    file1 = 'webapp\\static\\tmp-img\\00101001.png'
    file2 = 'webapp\\static\\tmp-img\\00101001.png'
    img1 = image.load_img(file1, grayscale = True, 
            target_size=(height, width))
            
    img1 = image.img_to_array(img1)#, dim_ordering='tf')
            
    images.append(img1)
    
    img2 = image.load_img(file2, grayscale = True, 
            target_size=(height, width))
        
    img2 = image.img_to_array(img2)#, dim_ordering='tf')
            
    images.append(img2)
        
    return np.array(images)

        
def main1(args):
    global datagen
    global model
    dataset = 'Bengali'
    dataset = 'Hindi'
    dataset = 'dataset2'
    if dataset == 'Bengali':
    
        tot_writers = 9
        num_train_writers = 7
        num_valid_writers = 1
        
    elif dataset == 'Hindi':
        
        tot_writers = 9
        num_train_writers = 7
        num_valid_writers = 1
        
    elif dataset == 'GPDS300':
    
        tot_writers = 300
        num_train_writers = 240
        num_valid_writers = 30
        
    elif dataset == 'GPDS960':
    
        tot_writers = 4000
        num_train_writers = 3200
        num_valid_writers = 400
        
    elif dataset == 'CEDAR1':
    
        tot_writers = 55
        num_train_writers = 45
        num_valid_writers = 5
    elif dataset == 'dataset2':
    
        tot_writers = 12
        num_train_writers = 10
        num_valid_writers = 1
    
    num_test_writers = tot_writers - (num_train_writers + num_valid_writers)
    
    
    # parameters
    batch_sz = 128
    nsamples = 276 
    img_height = 155
    img_width = 220
    featurewise_center = False
    featurewise_std_normalization = True
    zca_whitening = False
    nb_epoch = 20    
    input_shape=(img_height, img_width, 1)

    # initialize data generator   
    datagen = SignatureDataGenerator(dataset, tot_writers, num_train_writers, 
        num_valid_writers, num_test_writers, nsamples, batch_sz, img_height, img_width,
        featurewise_center, featurewise_std_normalization, zca_whitening)
    
    # data fit for std
    X_sample = read_signature_data(dataset, int(0.5*tot_writers), height=img_height, width=img_width)
    datagen.fit(X_sample)
    #del X_sample
    print('creating the network')
    # network definition
    base_network = create_base_network_signet(input_shape)
    
    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))
    
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    print ("hello:")
    model = Model(input=[input_a, input_b], output=distance)
    
    # compile model
    rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
    adadelta = Adadelta()
    model.compile(loss=contrastive_loss, optimizer=rms)
    # display model
#    plot(model, show_shapes=True)
#    sys.exit()     
    
    # callbacks
    fname = 'D:\\cocoapp-master\\cocoapp-master\\model\\dataset2.hdf5'
    #fname = 'D:\\cocoapp-master\\cocoapp-master\\model\\weights_Bengali.hdf5'
    #fname = 'D:\\cocoapp-master\\cocoapp-master\\model\\weights_Hindi.hdf5'
#     fname = '/home/sounak/Desktop/weights_GPDS300.hdf5'
    #checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
#    tbpointer = TensorBoard(log_dir='/home/adutta/Desktop/Graph', histogram_freq=0,  
#          write_graph=True, write_images=True)
    #print int(datagen.samples_per_valid)
    # print datagen.samples_per_train
    # print int(datagen.samples_per_valid)
    # print int(datagen.samples_per_test)
    # sys.exit()
    # train model   
    #model.fit_generator(generator=datagen.next_train(), steps_per_epoch=100,samples_per_epoch=datagen.samples_per_train, nb_epoch=nb_epoch,
                        #validation_data=datagen.next_valid(), nb_val_samples=int(datagen.samples_per_valid),callbacks=[checkpointer])   # KERAS 1

    # model.fit_generator(generator=datagen.next_train(), steps_per_epoch=960, epochs=nb_epoch,
    #                     validation_data=datagen.next_valid(), validation_steps=120, callbacks=[checkpointer])  # KERAS 2
    # load the best weights for test
    model.load_weights(fname)
    print (fname)
    print ('Loading the best weights for testing done...') 
   
    #print (datagen.next_test())
    #tr_pred = model.predict_generator(generator=datagen.next_train(), val_samples=int(datagen.samples_per_train))
    '''
    te_pred = model.predict(datagen.next_test())
    print (te_pred)
    if(te_pred > 0.5):
        te_pred = 0
    else:
        te_pred = 1
    
    
    print (te_pred)
    '''
    '''
#    tr_acc = compute_accuracy_roc(tr_pred, datagen.train_labels)
#    te_acc = compute_accuracy_roc(te_pred, datagen.test_labels)
    
#    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    #print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    '''
# Main Function    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Signature Verification')
    # required training parameters
    parser.add_argument('--dataset', '-ds', action='store', type=str, default='dataset2',
                  help='Please mention the database.')
    # required tensorflow parameters
    parser.add_argument('--epoch', '-e', action='store', type=int, default=20, 
                  help='The maximum number of iterations. Default: 20')
    parser.add_argument('--num_samples', '-ns', action='store', type=int, default=276, 
                  help='The number of samples. Default: 276')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=138,
                  help='The mini batch size. Default: 138')
    args = parser.parse_args()
# print args.dataset, args.epoch, args.num_samples, args.batch_size
#    sys.exit()
    main(args)
    
import fnmatch    
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
           # print (basename)
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


import sqlite3


def createconnection():
    con = sqlite3.connect('test7.db')
    cursor = con.cursor()
    return cursor

    
def insertTable(Signatureid, filename,picture_file):
    insert_query = """INSERT INTO dataset (ID, fileName,file) VALUES(?,?, ?)"""
    c = createconnection()
    with open(picture_file, 'rb') as picture_file:
        ablob = picture_file.read()
        c.execute(insert_query, (Signatureid, filename, ablob))
    c.connection.commit()
    
    
def get_file_from_db(customer_id):
	cursor = createconnection()
	select_fname = """SELECT file,fileName from dataset where ID = ?"""
	cursor.execute(select_fname, (customer_id,))
	item = cursor.fetchall()
	cursor.connection.commit()
	return item


    
def getpredictions(imagename, customer_id):
    print (customer_id)
    #fullName = 'D:\\cocoapp-master\\cocoapp-master\\model\\Bengali\\' + id 
    #pattern = '*-G-*'
    #fullName = 'D:\\cocoapp-master\\cocoapp-master\\model\\Hindi\\' + customer_id 
    #pattern = '*-G-*'
    #fullName = 'D:\\cocoapp-master\\cocoapp-master\\model\\dataset3\\' 
    #pattern = id + '??' + id + '.png'
    #files = []
    
    testFile = os.path.join(app.config['UPLOAD_FOLDER'], imagename)
    #for filename in find_files(fullName, pattern):
        #files.append(filename)
    
    items = get_file_from_db(customer_id)
        #f = 'D:\\cocoapp-master\\cocoapp-master\\cocoapp\\static\\tmp-img\\H-S-1-G-03.jpg'
        #testFile = 'D:\\cocoapp-master\\cocoapp-master\\cocoapp\\static\\tmp-img\\H-S-1-G-03.jpg'
        #ablob = get_files_from_db(id)
    te_pred = 0
    for file in items:
        try:
            originalFile = os.path.join(app.config['UPLOAD_FOLDER'], file[1])
            with open(originalFile, 'wb') as output_file:
                output_file.write(file[0])
                print (originalFile)
                print (testFile)
                te_pred = model.predict(datagen.next_test(file1 = originalFile,file2 = testFile))
                print (file[1], testFile, te_pred)
                if(te_pred == 0.0):
                    te_pred = 1
                    break
                else:
                    te_pred = 0
            #print (file[1])
        except:
            print('error')
    return te_pred
        
        

    
    