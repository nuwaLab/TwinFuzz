import argparse
import tensorflow as tf
import scipy.io as io
import numpy as np
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam



def VGG16(input_shape=(32, 32, 3), num_classes=10):
    input_tensor = layers.Input(shape=input_shape)

    # we define the block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # we define the block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # we define the block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # we define the block4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # we define the block5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # FLATTEN layer
    x = layers.Flatten(name='flatten')(x)

    # FC Layers
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)

    # Output layer (adjust to the number of classes in CIFAR-10)
    x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create model
    model = models.Model(inputs=input_tensor, outputs=x)

    return model

# Example usage:
# vgg16_cifar10 = VGG16(input_shape=(32, 32, 3), num_classes=10)
# vgg16_cifar10.summary()


# Thanks for the DistXplore get_model code
def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
    conv = layers.Conv2D(num_filters, 
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=keras.regularizers.l2(1e-4))
    
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    return x


def resnet_20(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 40)')
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    
    inputs = layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block==0:
                strides = 2
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = layers.Activation('relu')(x)
        num_filters *= 2
        
    x = layers.AveragePooling2D(pool_size=8)(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, 
                    kernel_initializer='he_normal')(y)
    outputs = layers.Activation('softmax')(outputs)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# model = resnet_20(input_shape = (32,32,3), depth=20)
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
# model.summary()


# Please follow the SVHN Webpage to download the dataset

def load_svhn():
    # Change the file path to the one which you used to store SVHN dataset
    x_train = io.loadmat('/data/mwt/DiffRobOT/SVHN/SVHN_data/train_32x32.mat')['X'] #73257
    y_train = io.loadmat('/data/mwt/DiffRobOT/SVHN/SVHN_data/train_32x32.mat')['y']

    x_test = io.loadmat('/data/mwt/DiffRobOT/SVHN/SVHN_data/test_32x32.mat')['X'] # 26032
    y_test = io.loadmat('/data/mwt/DiffRobOT/SVHN/SVHN_data/test_32x32.mat')['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    return (x_train, y_train), (x_test, y_test)





if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Access x_train, y_train; x_test, y_test
    x_train, y_train = load_svhn()[0]
    x_test, y_test = load_svhn()[1]
    # Conduct basic Normalization:
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Change the label 10 to label 0 as indicated in the dataset
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    # Prepare one-hot Encoder to the corresponding labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test =  keras.utils.to_categorical(y_test, 10)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices=['vgg16', 'resnet-20','vgg16-test','resnet-20-test'], help='models for training and testing')
    args = parser.parse_args()
    if args.m == 'vgg16':
        VGG16 = VGG16()
        VGG16.summary()
        VGG16.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 0.0001), metrics=['accuracy'])
        VGG16.fit(x_train, y_train, epochs=20, batch_size = 64)
        VGG16.save('./VGG16_SVHN.h5')
    if args.m == 'resnet-20':
        Resnet20 = resnet_20(input_shape = (32,32,3), depth = 20)
        Resnet20.summary()
        Resnet20.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        Resnet20.fit(x_train, y_train, epochs=10, batch_size = 128)
        Resnet20.save('./Resnet_20_SVHN.h5')
    if args.m == 'resnet-20-test':
        Resnet20 = load_model('./Resnet_20_SVHN.h5')
        Resnet20.summary()
        test_loss, test_accuracy = Resnet20.evaluate(x_test, y_test, verbose = 2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy} ")
    if args.m == 'vgg16-test':
        VGG16 = load_model('./VGG16_SVHN.h5')
        VGG16.summary()
        test_loss, test_accuracy = VGG16.evaluate(x_test, y_test, verbose = 2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")