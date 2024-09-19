import argparse
import tensorflow as tf
from tensorflow import keras
from models import LeNet5
from models import LeNet4
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# Reshape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0
# One-Hot Label
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices=['lenet4', 'lenet5', 'lenet4-test', 'lenet5-test'], help='models for training')
    args = parser.parse_args()
    if args.m == 'lenet4':
        lenet4 = LeNet4()
        lenet4.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        lenet4.fit(x_train, y_train, epochs=10, batch_size = 64)
        lenet4.save('./LeNet4_Fashion_MNIST.h5')
    if args.m == 'lenet5':
        leNet5 = LeNet5()
        leNet5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        leNet5.fit(x_train, y_train, epochs=10, batch_size=64)
        leNet5.evaluate(x_test, y_test)
        leNet5.save('./LeNet5_Fashion_MNIST.h5')
    if args.m == 'lenet4-test':
        lenet4 = load_model('./LeNet4_Fashion_MNIST.h5')
        lenet4.summary()
        test_loss, test_accuracy = lenet4.evaluate(x_test, y_test, verbose = 2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")
    if args.m == 'lenet5-test':
        lenet5 = load_model('./LeNet5_Fashion_MNIST.h5')
        lenet5.summary()
        test_loss, test_accuracy = lenet5.evaluate(x_test, y_test, verbose = 2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")