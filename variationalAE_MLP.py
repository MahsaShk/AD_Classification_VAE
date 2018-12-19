''' This scipt uses a deep variational autoencoder (VAE) to learn a latent feature
    representation from the low-level features and trains a multi-layer perceptron 
    (MLP) for two class classification purpose.
'''
import numpy as np
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.callbacks import (EarlyStopping, 
                             LearningRateScheduler, 
                             ModelCheckpoint,
                             History)
from keras.regularizers import l2
import time
import glob
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support


if __name__ == '__main__':
    
    # Read dataset
    print("-------------Read data------------- ")
    pathNC = './data/NC.csv' # Normal data
    pathp = './data/AD.csv'  # Patient data
    n_classes = 2


    # Extract mesh coordinates from csv files
    dataNC = np.genfromtxt(pathNC,delimiter=',')
    datap = np.genfromtxt(pathp,delimiter=',')
    print(" normal data is of size ", dataNC.shape )
    print(" patient data is of size ", datap.shape )
    data = np.vstack((dataNC,datap))

    # Define target
    nb_s_NC = dataNC.shape[0]
    nb_s_p = datap.shape[0]   
    target_NC = 1*np.ones((nb_s_NC,1))
    target_p = 2*np.ones((nb_s_p,1))
    target = np.vstack((target_NC,target_p))
    target = target.astype(int)

    np.savetxt("./data/target.csv", target, delimiter=",")
    np.savetxt("./data/data.csv", data, delimiter=",")
    print("---------data and target are created------- ")

    print("------------permutation starts-------------")
    X,  y = shuffle(data, target, random_state=0) # shuffles the rows. random_state == seed

    print("------------Normalization starts-----------")
    scaler = preprocessing.StandardScaler().fit(X) 
    X_scaled = scaler.transform(X) 

    print("-----------train/test split starts---------")
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42) 
    
    print("-------------setting network param---------")
    batch_size = 28
    original_dim = data.shape[1] 
    latent_dim = 128 
    intermediate_dim1 = 512
    epsilon_std = 0.01
    nb_epoch = 100 
    kl_coef = 0

    print("-------------define network model---------")
    #Define VAE
    x = Input(shape=(original_dim,), name='input')
    h1 = Dense(intermediate_dim1, activation='relu', kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.05))(x)
    h1_do = Dropout(0.5)(h1)
    z_mean = Dense(latent_dim)(h1_do)
    z_log_std = Dense(latent_dim)(h1_do)
    
    def sampling(args):  
        z_mean, z_log_std = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(z_log_std) * epsilon
    
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_std])
    z_do = Dropout(0.5)(z)    
    decoder_h = Dense(intermediate_dim1, activation='relu', kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.05))
    h_decoded = decoder_h(z_do)
    
    decoder_mean = Dense(original_dim, activation='relu', name='vae_output', kernel_initializer='lecun_uniform',kernel_regularizer=l2(0.05))
    x_decoded_mean = decoder_mean(h_decoded)

    #Define MLP
    def combine_mean_std(args):
        z_mean, z_log_std = args
        return z_mean + K.exp(z_log_std)
    
    MLP_in = Lambda(combine_mean_std, output_shape=(latent_dim,))([z_mean, z_log_std])
    MLP_in_do = Dropout(0.5)(MLP_in)
    MLP1 = Dense (latent_dim, activation='relu', kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.05))(MLP_in_do)#, W_regularizer=l2(0.01)
    sftmx = Dense(n_classes, activation='softmax', name='classification_out')(MLP1)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean) #
        kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
        return K.mean(xent_loss + kl_loss)


    vae = Model(input=x, output=[x_decoded_mean, sftmx])
    vae.compile(optimizer='rmsprop', loss={'vae_output': vae_loss, 'classification_out': 'categorical_crossentropy'}, metrics={'classification_out': 'accuracy'})
    
    #change learning rate from 0.0001 to 0.00001 in a logarithmic way
    LearningRate = np.logspace(-5, -6, num=nb_epoch)
    LearningRate = LearningRate.astype('float32')
    K.set_value(vae.optimizer.lr, LearningRate[0])    
    def scheduler(epoch):
        K.set_value(vae.optimizer.lr, LearningRate[epoch-1])
        return float(K.get_value(vae.optimizer.lr))

    change_lr = LearningRateScheduler(scheduler)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    filepath="./data/weights_resunet_"+ timestr + ".hdf5"
    early_stopping = EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto') 
    checkpointer = ModelCheckpoint(filepath="./data/weights_resunet_"+ timestr + ".hdf5", verbose=1, save_best_only=True)    
    print ("-------------network fit starts----------------")
    from keras.utils.np_utils import to_categorical
    history = History()
    fitlog = vae.fit({'input': x_train}, {'vae_output': x_train, 'classification_out': to_categorical(y_train-1)},
            shuffle=True,
            epochs=100,
            batch_size=batch_size,
            validation_split=0.2, validation_data=None,verbose=1, callbacks=[history])
    
    # -----------evaluate --------------------------
    print ("\n\n---------evaluating on test set starts--------\n ")
    score = vae.evaluate({'input': x_test}, {'vae_output': x_test, 'classification_out': to_categorical(y_test-1)}, batch_size=batch_size)
    print(vae.metrics_names)
    print (" ", score)
    print ("\nClassification accuracy on test set is : ", score[3])

    
 
