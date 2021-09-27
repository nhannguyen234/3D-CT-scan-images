import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import *
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling3D, Flatten, Dense, MaxPooling3D, Conv3D,\
                                    LeakyReLU, BatchNormalization, Add, ReLU, Dropout, ZeroPadding3D, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers.experimental import preprocessing

def res_block(x, kernel_size, units):
    fx = BatchNormalization()(x)
    fx = ReLU()(fx)
    fx = Conv3D(units, kernel_size, kernel_regularizer=l2(1e-5), padding='same')(fx)
    fx = BatchNormalization()(fx)
    fx = ReLU()(fx)
    fx = Conv3D(units, kernel_size, kernel_regularizer=l2(1e-5), padding='same')(fx)
    out = Add()([x,fx])
    return out

def resnet_34(inputs, units):
    y = Conv3D(units, (3,3,3),kernel_regularizer=l2(1e-5), padding='same')(inputs)
    y = res_block(y, (3,3,3), units)
    y = res_block(y, (3,3,3), units)
    y = Conv3D(units*2, (3,3,3),kernel_regularizer=l2(1e-5), padding='same')(y)
    y = res_block(y, (3,3,3), units*2)
    y = res_block(y, (3,3,3), units*2)
    y = res_block(y, (3,3,3), units*2)
    y = Conv3D(units*3, (3,3,3),kernel_regularizer=l2(1e-5), padding='same')(y)
    y = res_block(y, (3,3,3), units*3)
    y = res_block(y, (3,3,3), units*3)
    y = res_block(y, (3,3,3), units*3)
    y = res_block(y, (3,3,3), units*3)
    y = res_block(y, (3,3,3), units*3)
    y = Conv3D(units*4, (3,3,3),kernel_regularizer=l2(1e-5), padding='same')(y)
    y = res_block(y, (3,3,3), units*4)
    outputs = res_block(y, (3,3,1), units*4)
    return outputs

def halve_block(inputs, units, kernel_size):
    x = Conv3D(units, kernel_size, activation='relu', padding='valid')(inputs)
    x = BatchNormalization()(x)
    out = MaxPooling3D((2,2,2))(x)
    return out

def repeat_block(inputs, units, z_p):
    y = ZeroPadding3D(z_p)(inputs)
    y = halve_block(y, units, (3,3,3))
    out = halve_block(y, units*2, (3,3,3))
    return out

def build_model(image_size, depth_size):
    inputs = Input(shape=(image_size, image_size, depth_size, 1))
    
#     x = img_augmentation(inputs)
    # block 1
    shortcut_1 = Conv3D(64, (3,3,3), strides=(2,2,2))(inputs)
    shortcut_1 = BatchNormalization()(shortcut_1)
    shortcut_1 = ReLU()(shortcut_1)
    shortcut_1 = MaxPooling3D((2,2,2))(shortcut_1)
    
    y1 = repeat_block(inputs, 32, (1,1,1))
    y2 = repeat_block(inputs, 64, (1,1,1))
    
    add_layer1 = Add()([shortcut_1, y1])
    out_block1 = Concatenate()([add_layer1, y2])
    
    # block 2 resnet 34
#     y = resnet_34(out_block1, 64)
#     out_block2 = ReLU()(y)
    
    #block 3
    shortcut_3 = Conv3D(64, (3,3,3), strides=(2,2,2))(out_block1)
    shortcut_3 = BatchNormalization()(shortcut_3)
    shortcut_3 = ReLU()(shortcut_3)
    shortcut_3 = MaxPooling3D((2,2,2))(shortcut_3)

    y3 = repeat_block(out_block1, 32, (3,3,2))
    y4 = repeat_block(out_block1, 64, (3,3,2))
    
    add_layer2 = Add()([shortcut_3, y3])
    out_block3 = Concatenate()([add_layer2, y4])
    
    y = BatchNormalization()(out_block3)
    y = ReLU()(y)
    
    #fully connected
    y = GlobalAveragePooling3D()(y)
    y = Dense(272, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = ReLU()(y)
    
    outputs = Dense(1, activation='sigmoid')(y)
    return Model(inputs, outputs)

def compile_model(model):
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    acc = tf.keras.metrics.BinaryAccuracy()
    auc = tf.keras.metrics.AUC()
    model.compile(optimizer=opt,
                 loss = 'binary_crossentropy',
                 metrics=[auc, acc])

# def scheduler(epoch, lr):
#     if epoch <= 12:
#         return 0.00001
#     else:
#         return lr*np.math.exp(-1.5)

# class epoch_end(Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if logs.get('val_binary_accuracy') > 0.9:
#             print(' Accuracy reached 0.9')
#             self.model.stop_training=True