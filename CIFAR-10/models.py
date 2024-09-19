from tensorflow import keras
from tensorflow.keras import layers, models
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf



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


def inception_block(x, filters):
    # Now corresponds to 1*1 convolution branch
    inception_a = layers.Conv2D(filters[0], (1,1), padding = 'same', activation = 'relu')(x)

    # Then, there exists 1*1 convolution followed by 3*3 convolution branch
    inception_b = layers.Conv2D(filters[1], (1,1), padding = 'same', activation = 'relu')(x)
    inception_b = layers.Conv2D(filters[2], (3,3), padding = 'same', activation = 'relu')(inception_b)

    # 5*5 convolution branch following 1*1
    inception_c = layers.Conv2D(filters[3], (1,1), padding = 'same', activation ='relu')(x)
    inception_c = layers.Conv2D(filters[4], (5,5), padding = 'same', activation ='relu')(inception_c)

    # Maxpooling Inception Branch
    branch_pooling = layers.MaxPooling2D((3,3), strides =(1,1), padding = 'same')(x)
    branch_pooling = layers.Conv2D(filters[5], (1,1), padding = 'same', activation = 'relu')(branch_pooling)

    # Concatenate the corresponding branches
    x = layers.concatenate([inception_a, inception_b, inception_c, branch_pooling], axis = -1)

    return x 

def auxiliary_classifier(x, num_classes):
    au = layers.AveragePooling2D((3,3), strides=2, padding = 'valid')(x)
    au = layers.Conv2D(128,(1,1), padding = 'same', activation = 'relu')(au)
    au = layers.Flatten()(au)
    au = layers.Dense(1024, activation = 'relu')(au)
    #
    au = layers.Dropout(0.7)(au)
    au = layers.Dense(num_classes, activation = 'softmax')(au)
    # Return our auxiliary_classifier
    return au


def GoogLeNet(input_shape = (32,32,3), num_classes = 10):

    # input into the corrresponding model
    input_tensor = layers.Input(shape = input_shape)

    # Initialize the model
    x = layers.Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(input_tensor)
    x = layers.MaxPooling2D((3,3), strides = (2,2), padding = 'same')(x)

    x = layers.Conv2D(64,(1,1), padding = 'same', activation = 'relu')(x)
    x = layers.Conv2D(192, (3,3), padding = 'same', activation = 'relu')(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding = 'same')(x)

    # Then, we fit in the Inception Branch
    x = inception_block(x, [64,96,128,16,32,32])
    x = inception_block(x, [128,128,192,32,96,64])
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

    x = inception_block(x, [192,96,208,16,48,64])

    # Put in the auxiliary_classifier
    auxiliary_1 = auxiliary_classifier(x, num_classes)

    x = inception_block(x,[160,112,224,24,64,64])
    x = inception_block(x, [128,128,256,24,64,64])
    x = inception_block(x, [112,144,288,32,64,64])

    auxiliary_2 = auxiliary_classifier(x, num_classes)

    x = inception_block(x, [256,160,320,32,128,128])
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding = 'same')(x)

    x = inception_block(x, [256,160,320,32,128,128])
    x = inception_block(x, [384,192,384,48,128,128])

    # Finally, we add global average pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(num_classes, activation = 'softmax')(x)

    # pass through a auxiliary classifier branch
    model = models.Model(inputs = input_tensor, outputs = [x, auxiliary_1, auxiliary_2])
    return model

# model = GoogLeNet(input_shape = (32,32,3), num_classes = 10)
# model.compile(optimizer='adam', 
#               loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], 
#               loss_weights=[1, 0.3, 0.3], 
#               metrics=['accuracy'])
# model.summary()


def inceptionv3_block(x, filters):

    # The first inception block inception_a
    inception_a = layers.Conv2D(filters[0], (1,1), padding = 'same', activation = 'relu')(x)

    # The second inception block inception_b
    inception_b = layers.Conv2D(filters[1], (1,1), padding = 'same', activation = 'relu')(x)
    inception_b = layers.Conv2D(filters[2], (3,3), padding = 'same', activation = 'relu')(inception_b)

    # Two 3*3 convolution branch in the block
    inception_c = layers.Conv2D(filters[3], (1,1), padding = 'same', activation = 'relu')(x)
    inception_c = layers.Conv2D(filters[4], (3,3), padding = 'same', activation = 'relu')(inception_c)
    inception_c = layers.Conv2D(filters[4], (3,3), padding = 'same', activation = 'relu')(inception_c)

    # Maxpooling and convolution
    inception_d = layers.MaxPooling2D((3,3), strides = (1,1), padding = 'same')(x)
    inception_d = layers.Conv2D(filters[5], (1,1), padding = 'same', activation = 'relu')(inception_d)

    # Concatenate branch
    x = layers.concatenate([inception_a, inception_b, inception_c, inception_d], axis = -1)

    return x

def inception_v3(input_shape = (32,32,3), num_classes = 10):
    input_tensor = layers.Input(shape = input_shape)

    # Initial convolutional layers and pooling layers.
    x = layers.Conv2D(32, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(input_tensor)
    x = layers.Conv2D(32, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
    x = layers.Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
    x = layers.MaxPooling2D((3,3), strides = (2,2), padding = 'same')(x)
    x = layers.Conv2D(80, (1,1), padding = 'same', activation = 'relu')(x)
    x = layers.Conv2D(192, (3,3), padding = 'same', activation = 'relu')(x)
    x = layers.MaxPooling2D((3,3), strides = (2,2), padding = 'same')(x)

    # Go through the inception block
    x = inceptionv3_block(x, [64, 48, 64, 64, 96, 32])
    x = inceptionv3_block(x, [64, 48, 64, 64, 96, 64])
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding = 'same')(x)

    x = inceptionv3_block(x, [128, 96, 128, 96, 128, 128])
    x = inceptionv3_block(x, [160, 128, 160, 128, 192, 128])
    x = inceptionv3_block(x, [192, 160, 192, 192, 256, 160])
    x = layers.MaxPooling2D((3,3), strides = (2,2), padding = 'same')(x)
    x = inceptionv3_block(x, [256, 160, 320, 160, 320, 160])

    # Add Global Average Pooling layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation = 'softmax')(x)

    return models.Model(input_tensor, x)


# model = inception_v3(input_shape = (32,32,3), num_classes = 10)
# model.summary()


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





# Inception_Resnet_v2 is partially derived from the tensorflow keras team inception_resnet_v2 source code:
# https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/inception_resnet_v2.py#L16-L243

def conv2d_bn(x, filters, kernel_size, strides=1, padding="same", activation="relu", use_bias=False, name=None):

    """Utility function to apply conv + BN.

    Args:
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'`
            for the activation and `name + '_bn'` for the batch norm layer.

    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x

# Custom scaling layer for residual blocks
class CustomScaleLayer(layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return inputs[0] + inputs[1] * self.scale
    
    # To save the model, define get_config function is necessary
    def get_config(self):
        config = super().get_config()
        config.update({
            'scale':self.scale,
            })
        return config


# Inception-ResNet block
def inception_resnet_block(x, scale, block_type, block_idx, activation="relu"):
    if block_type == "block35":
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == "block17":
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == "block8":
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError(f"Unknown block type: {block_type}")

    mixed = layers.Concatenate()(branches)
    up = conv2d_bn(mixed, x.shape[-1], 1, activation=None, use_bias=True)
    x = CustomScaleLayer(scale)([x, up])
    if activation:
        x = layers.Activation(activation)(x)
    return x

# Reduction blocks
def reduction_A(x):
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding="valid")
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding="valid")
    branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
    return layers.Concatenate()([branch_0, branch_1, branch_pool])

def reduction_B(x):
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding="valid")
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding="valid")
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding="valid")
    branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
    return layers.Concatenate()([branch_0, branch_1, branch_2, branch_pool])

# Inception-ResNet-v2 Model for CIFAR-10
def InceptionResNetV2(input_shape=(32, 32, 3), classes=10, include_top=True):
    inputs = layers.Input(shape=input_shape)

    # Stem block
    x = conv2d_bn(inputs, 32, 3)
    x = conv2d_bn(x, 32, 3)
    x = conv2d_bn(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Inception-ResNet-A blocks
    for _ in range(5):
        x = inception_resnet_block(x, scale=0.17, block_type="block35", block_idx=0)

    # Reduction-A
    x = reduction_A(x)

    # Inception-ResNet-B blocks
    for _ in range(10):
        x = inception_resnet_block(x, scale=0.1, block_type="block17", block_idx=0)

    # Reduction-B
    x = reduction_B(x)

    # Inception-ResNet-C blocks
    for _ in range(5):
        x = inception_resnet_block(x, scale=0.2, block_type="block8", block_idx=0)

    # Final Convolution Block
    x = conv2d_bn(x, 1536, 1)

    # Classification block
    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(classes, activation="softmax")(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)

    model = models.Model(inputs, x, name="inception_resnet_v2_cifar10")
    return model

# Establish the corresponding model
# model = InceptionResNetV2(input_shape=(32, 32, 3), classes=10, include_top=True)
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Access x_train, y_train; x_test, y_test
    x_train, y_train = load_cifar10()[0]
    x_test, y_test = load_cifar10()[1]
    # Reshape
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    # Normalization
    x_train, x_test = x_train/255.0, x_test/ 255.0
    # One-hot label
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # print(x_train.shape) #(50000, 32, 32, 3)
    # print(x_test.shape) #(10000, 32, 32, 3)
    # print(y_train.shape) # (50000, 10) to_categorical
    # print(y_test.shape) # (10000, 10) to_categorical

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices=['vgg16', 'resnet-20','vgg16-test','resnet-20-test','Inception-ResNet-V2','Inception-ResNet-V2-test',
                                       'GoogLenet', 'GoogLenet-test','Inception_v3','Inception_v3-test' ], help='models for training and testing')
    args = parser.parse_args()

    if args.m == 'vgg16':
        VGG16 = VGG16()
        VGG16.summary()
        VGG16.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 0.0001), metrics=['accuracy'])
        VGG16.fit(x_train, y_train, epochs=20, batch_size = 64)
        VGG16.save('./VGG16_CIFAR10.h5')

    if args.m == 'resnet-20':
        Resnet20 = resnet_20(input_shape = (32,32,3), depth = 20)
        Resnet20.summary()
        Resnet20.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics=['accuracy'])
        Resnet20.fit(x_train, y_train, epochs=20, batch_size = 64)
        Resnet20.save('./Resnet_20_CIFAR10.h5')

    if args.m == 'resnet-20-test':
        Resnet20 = load_model('./Resnet_20_CIFAR10.h5')
        Resnet20.summary()
        test_loss, test_accuracy = Resnet20.evaluate(x_test, y_test, verbose = 2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy} ")

    if args.m == 'vgg16-test':
        VGG16 = load_model('./VGG16_CIFAR10.h5')
        VGG16.summary()
        test_loss, test_accuracy = VGG16.evaluate(x_test, y_test, verbose = 2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")

    if args.m == 'Inception-ResNet-V2':
        InRes_v2 = InceptionResNetV2(input_shape = (32,32,3), classes=10, include_top=True) 
        InRes_v2.summary()
        InRes_v2.compile(optimizer=Adam(learning_rate = 0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
        InRes_v2.fit(x_train, y_train, epochs=20, batch_size = 64)
        InRes_v2.save('./InRes_v2_CIFAR10.h5')

    if args.m == 'Inception-ResNet-V2-test':
        InRes_v2 = load_model('./InRes_v2_CIFAR10.h5', custom_objects={'CustomScaleLayer': CustomScaleLayer})
        InRes_v2.summary()
        test_loss, test_accuracy = InRes_v2.evaluate(x_test, y_test, verbose = 2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")

    if args.m == 'GoogLenet':
        GoogLenet = GoogLeNet(input_shape = (32,32,3), num_classes = 10)
        GoogLenet.summary()
        GoogLenet.compile(optimizer=Adam(learning_rate = 0.0001), 
              loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], 
              loss_weights=[1, 0.3, 0.3], 
              metrics=['accuracy'])
        GoogLenet.fit(x_train, y_train, epochs = 20, batch_size = 64)
        GoogLenet.save('./GoogLenet_CIFAR10.h5')

    if args.m == 'GoogLenet-test':
        GoogLenet = load_model('./GoogLenet_CIFAR10.h5')
        main_output = GoogLenet.evaluate(x_test, y_test, verbose = 2)
        test_loss = main_output[0]
        test_accuracy = main_output[-1]
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")
    
    if args.m == 'Inception_v3':
        Inv3 = inception_v3(input_shape = (32,32,3), num_classes = 10)
        Inv3.summary()
        Inv3.compile(optimizer = Adam(learning_rate = 0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
        Inv3.fit(x_train, y_train, epochs=20, batch_size = 64)
        Inv3.save('./Inv3_CIFAR10.h5')

    if args.m == 'Inception_v3-test':
        Inv3 = load_model('./Inv3_CIFAR10.h5')
        Inv3.summary()
        test_loss, test_accuracy = Inv3.evaluate(x_test, y_test, verbose = 2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")



