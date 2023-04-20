import tensorflow as tf
from tensorflow.keras.layers import Conv2D, TimeDistributed, Dropout, Input, Dense, \
    BatchNormalization, GRU, Layer, Flatten, MaxPooling2D, concatenate, Lambda

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import ConvLSTM2D

def get_unet_lstm_keras(slices=3):
    input_shape = (slices, 256, 256, 1)
    input_l = layers.Input(shape=(input_shape))
    #Contraction path
    x =  (layers.TimeDistributed(layers.Conv2D( 16, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu'))) (input_l)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    conv1 = layers.TimeDistributed( layers.Conv2D( 16, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu' ) ) (x)
    conv1=layers.TimeDistributed(layers.BatchNormalization())(conv1)
    x=layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2,2)))(conv1)
    x = layers.TimeDistributed( layers.Conv2D( 32, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal',activation='relu' ) ) (x)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    conv2 = layers.TimeDistributed( layers.Conv2D( 32, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu' ) ) (x)
    conv2=layers.TimeDistributed(layers.BatchNormalization())(conv2)
    x=layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2,2)))(conv2)
    x = layers.TimeDistributed( layers.Conv2D( 64, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu' ) ) (x)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    conv3 = layers.TimeDistributed( layers.Conv2D( 64, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu' ) ) (x)
    conv3=layers.TimeDistributed(layers.BatchNormalization())(conv3)
    x=layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2,2)))(conv3)
    x = layers.TimeDistributed( layers.Conv2D( 128, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu' ) ) (x)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    conv4 = layers.TimeDistributed( layers.Conv2D( 128, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu')) (x)
    conv4=layers.TimeDistributed(layers.BatchNormalization())(conv4)
    x=layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2,2)))(conv4)
    x = layers.TimeDistributed( layers.Conv2D( 256, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu' ) ) (x)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    conv5 = layers.TimeDistributed( layers.Conv2D( 256, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal',activation='relu' ) ) (x)
    conv5=layers.TimeDistributed(layers.BatchNormalization())(conv5)
    # LSTM component
    x=layers.ConvLSTM2D(256,kernel_size=(3,3),padding='same',strides=(1,1),return_sequences=True,recurrent_dropout=0.2)(conv5)
    #Expansive path
    up1 = layers.TimeDistributed( layers.Conv2DTranspose(128,kernel_size=(3,3),padding='same',strides=(2,2)))(x)
    concat1 = layers.concatenate([up1, conv4])
    x = layers.TimeDistributed( layers.Conv2D( 128, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu' ) ) (concat1)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed( layers.Conv2D( 128, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu') ) (x)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    up2 = layers.TimeDistributed( layers.Conv2DTranspose( 64,kernel_size=(3,3),padding='same',strides=(2,2)))(x)
    concat2 = layers.concatenate([up2, conv3])
    x = layers.TimeDistributed( layers.Conv2D( 64, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu' ) ) (concat2)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed( layers.Conv2D( 64, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu' ) ) (x)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    up3 = layers.TimeDistributed( layers.Conv2DTranspose( 32,kernel_size=(3,3),padding='same',strides=(2,2)))(x)
    concat3 = layers.concatenate([up3, conv2])
    x = layers.TimeDistributed( layers.Conv2D( 32, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal',activation='relu') ) (concat3)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed( layers.Conv2D( 32, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal',activation='relu') ) (x)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    up4= layers.TimeDistributed( layers.Conv2DTranspose( 16,kernel_size=(3,3),padding='same',strides=(2,2)))(x)
    concat4 = layers.concatenate([up4, conv1])
    x = layers.TimeDistributed( layers.Conv2D( 16, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal', activation='relu') ) (concat4)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed( layers.Conv2D( 16, kernel_size=(3, 3),padding='same',strides=(1,1),kernel_initializer='he_normal',activation='relu') ) (x)
    x=layers.TimeDistributed(layers.BatchNormalization())(x)
    #LSTM component
    x=layers.ConvLSTM2D(16,kernel_size=(3,3),padding='same',strides=(1,1),return_sequences=True,recurrent_dropout=0.2)(x)
    #x=tf.expand_dims(x,axis=1)
    out = layers.Conv2D( 1, kernel_size=(1, 1),padding='same',strides=(1,1), activation='sigmoid')(x)
    model = models.Model(inputs=input_l, outputs=out)
    model.summary()

    return model
