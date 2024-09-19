from tensorflow import keras
from tensorflow.keras import layers, models

def resnet_block(input_tensor, filters, kernel_size = 3, stride = 1, conv_shortcut = False):
    """This resnet_block corresponds to the path shortcut"""
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, (1,1), strides = stride, padding = 'same')(input_tensor)
    else:
        shortcut = input_tensor
    
    x = layers.Conv2D(filters, kernel_size, strides = stride, padding = 'same', activation = 'relu')(input_tensor)
    # In Resnet, we have the batchnormalization layer
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding = 'same')(x)
    # Here, we have the batchnormalization layer
    x = layers.BatchNormalization()(x)

    # Add Residual Block shortcut
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

def ResNet20(input_shape = (32,32,3), num_classes = 10):
    """Here we define ResNet20 """
    input_tensor = layers.Input(shape = input_shape)

    # Then, we construct the initial layer
    x = layers.Conv2D(16,(3,3), padding='same', strides = 1, activation='relu')(input_tensor)
    x = layers.BatchNormalization()(x)

    # Stage1 >- 6 blocks, 16 filters
    for _ in range(6):
        x = resnet_block(x, 16)

    # Stage2 >- 6 blocks, 32 filters
    x = resnet_block(x, 32, stride = 2, conv_shortcut=True)
    for _ in range(5):
        x = resnet_block(x, 32)
    
    # Stage3 >- 6 blocks, 64 filters
    x = resnet_block(x, 64, stride = 2, conv_shortcut=True)
    for _ in range(5):
        x = resnet_block(x, 64)
    
    # Final Softmax layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation = 'softmax')(x)

    # We then create the corresponding model
    model = models.Model(inputs = input_tensor, outputs = x)

    return model

# resnet20 = ResNet20(input_shape=(32,32,3), num_classes=10)
# resnet20.summary()


def InceptionResNetV2(
    include_top=True,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=10,
    classifier_activation="softmax",
):
    """Instantiates the Inception-ResNet v2 architecture.

    Reference:
    - [Inception-v4, Inception-ResNet and the Impact of
       Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
      (AAAI 2017)

    This function returns a Keras image classification model,
    optionally loaded with weights pre-trained on ImageNet.

    For image classification use cases, see
    [this page for detailed examples](
      https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).

    Note: each Keras Application expects a specific kind of
    input preprocessing. For InceptionResNetV2, call
    `keras.applications.inception_resnet_v2.preprocess_input`
    on your inputs before passing them to the model.
    `inception_resnet_v2.preprocess_input`
    will scale input pixels between -1 and 1.

    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            `"imagenet"` (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)`
            (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`,
            and if no `weights` argument is specified.
        classifier_activation: A `str` or callable.
            The activation function to use on the "top" layer.
            Ignored unless `include_top=True`.
            Set `classifier_activation=None` to return the logits
            of the "top" layer. When loading pretrained weights,
            `classifier_activation` can only be `None` or `"softmax"`.

    Returns:
        A model instance.
    """

    # Determine proper input shape
    input_shape = (32,32,3)
    input_tensor = layers.Input(shape = input_shape)

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding="valid")
    x = conv2d_bn(x, 32, 3, padding="valid")
    x = conv2d_bn(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding="valid")
    x = conv2d_bn(x, 192, 3, padding="valid")
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == "channels_first" else 3
    x = layers.Concatenate(axis=channel_axis, name="mixed_5b")(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(
            x, scale=0.17, block_type="block35", block_idx=block_idx
        )

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding="valid")
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding="valid")
    branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name="mixed_6a")(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(
            x, scale=0.1, block_type="block17", block_idx=block_idx
        )

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding="valid")
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding="valid")
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding="valid")
    branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name="mixed_7a")(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(
            x, scale=0.2, block_type="block8", block_idx=block_idx
        )
    x = inception_resnet_block(
        x, scale=1.0, activation=None, block_type="block8", block_idx=10
    )

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name="conv_7b")

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras.src.ops.operation_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = keras.src.models.Functional(inputs, x, name="inception_resnet_v2")

    

    return model
    

def conv2d_bn(
    x,
    filters,
    kernel_size,
    strides=1,
    padding="same",
    activation="relu",
    use_bias=False,
    name=None,
):
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
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name,
    )(x)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == "channels_first" else 3
        bn_name = None if name is None else name + "_bn"
        x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(
            x
        )
    if activation is not None:
        ac_name = None if name is None else name + "_ac"
        x = layers.Activation(activation, name=ac_name)(x)
    return x


class CustomScaleLayer(Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config

    def call(self, inputs):
        return inputs[0] + inputs[1] * self.scale


def inception_resnet_block(x, scale, block_type, block_idx, activation="relu"):
    """Adds an Inception-ResNet block.

    Args:
        x: input tensor.
        scale: scaling factor to scale the residuals
            (i.e., the output of passing `x` through an inception module)
            before adding them to the shortcut
            branch. Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`,
            determines the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks are repeated many times
            in this network. We use `block_idx` to identify each
            of the repetitions. For example, the first
            Inception-ResNet-A block will have
            `block_type='block35', block_idx=0`, and the layer names
            will have a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block.

    Returns:
        Output tensor for the block.
    """
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
        raise ValueError(
            "Unknown Inception-ResNet block type. "
            'Expects "block35", "block17" or "block8", '
            "but got: " + str(block_type)
        )

    block_name = block_type + "_" + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == "channels_first" else 3
    mixed = layers.Concatenate(axis=channel_axis, name=block_name + "_mixed")(
        branches
    )
    up = conv2d_bn(
        mixed,
        x.shape[channel_axis],
        1,
        activation=None,
        use_bias=True,
        name=block_name + "_conv",
    )

    x = CustomScaleLayer(scale)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + "_ac")(x)
    return x