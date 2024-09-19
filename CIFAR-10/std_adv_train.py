import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.io as io
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from models import CustomScaleLayer
# ==== Configuration ====
name = "Resnet_20"

# Fast Gradient Sign Method
def fgsm_attack(model, inputs, labels, ep=0.3, isRand=True, randRate=1):
    
    in_cp = inputs.copy() # shallow copy, reference only
    target = tf.constant(labels)
    
    if isRand:
        inputs = np.random.uniform(-ep * randRate, ep * randRate, inputs.shape)
        inputs = np.clip(inputs, 0, 1)

    # check the input data format.
    inputs = tf.Variable(inputs, dtype = tf.float32)
    model.trainable = False
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    with tf.GradientTape() as tape:
        loss = keras.losses.categorical_crossentropy(target, model(inputs))
        grads = tape.gradient(loss, inputs)
    
    in_adv = inputs + ep * tf.sign(grads)
    in_adv = tf.clip_by_value(in_adv, clip_value_min=in_cp-ep, clip_value_max=in_cp+ep)
    in_adv = tf.clip_by_value(in_adv, clip_value_min=0, clip_value_max=1)

    idxs = np.where(np.argmax(model(in_adv), axis=1) != np.argmax(labels, axis=1))[0]
    print("[INFO] Success FGSM Attack Num:", len(idxs))

    in_adv, in_cp, target = in_adv.numpy()[idxs], in_cp[idxs], target.numpy()[idxs]
    in_adv, target = tf.Variable(in_adv), tf.constant(target)

    return in_adv.numpy(), target.numpy()


# Projected Gradient Descent
def pgd_attack(model, inputs, labels, step, ep=0.3, epochs=10, isRand=True, randRate=1):

    in_cp = inputs.copy()
    target = tf.constant(labels)

    if step == None:
        step = ep / 8

    if isRand:
        inputs = inputs + np.random.uniform(-ep * randRate, ep * randRate)
        inputs = np.clip(inputs, 0, 1)
    
    # Specify the datatype

    in_adv = tf.Variable(inputs, dtype = tf.float32)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(target, model(in_adv))
            grads = tape.gradient(loss, in_adv)
        
        in_adv.assign_add(step * tf.sign(grads))
        in_adv = tf.clip_by_value(in_adv, clip_value_min=in_cp-ep, clip_value_max=in_cp+ep)
        in_adv = tf.clip_by_value(in_adv, clip_value_min=0, clip_value_max=1)
        in_adv = tf.Variable(in_adv)

    idxs = np.where(np.argmax(model(in_adv), axis=1) != np.argmax(target, axis=1))[0]
    print("[INFO] Success PGD Attack Num:", len(idxs))

    in_adv, in_cp, target = in_adv.numpy()[idxs], in_cp[idxs], target.numpy()[idxs]
    in_adv, target = tf.Variable(in_adv), tf.constant(target)

    return in_adv.numpy(), target.numpy()

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # Conduct reshape processes
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    # Conduct Normalization
    x_train, x_test = x_train/255.0, x_test/255.0
    # One-hot Label Conversion
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, x_test, y_train, y_test


def gen_adv_samples_test(batch_size = 64):


    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    # Load the corresponding x_train, y_train, x_test, y_test from svhn_Dataset
    
    # For InRes_v2:
    # model = keras.models.load_model(f"./{name}_CIFAR10.h5", custom_objects={'CustomScaleLayer': CustomScaleLayer})
    model = keras.models.load_model(f"./{name}_CIFAR10.h5")
    
    


    x_train, x_test, y_train, y_test = load_cifar10()

    # Prepare a list to store adversarial samples and labels for attack
    adv_train_fgsm, adv_train_pgd = [], []
    label_train_fgsm, label_train_pgd = [], []

    adv_test_fgsm, adv_test_pgd = [], []
    label_test_fgsm, label_test_pgd = [], []

    # Process training data in batches
    for i in range(0, len(x_train), batch_size):
        x_train_batch = x_train[i:i + batch_size]
        y_train_batch = y_train[i:i + batch_size]

        # FGSM attack on training data
        advs, labels = fgsm_attack(model, x_train_batch, y_train_batch)
        adv_train_fgsm.append(advs)
        label_train_fgsm.append(labels)

        # PGD attack on training data
        advs, labels = pgd_attack(model, x_train_batch, y_train_batch, step=None)
        adv_train_pgd.append(advs)
        label_train_pgd.append(labels)

    # Process testing data in batches
    for i in range(0, len(x_test), batch_size):
        x_test_batch = x_test[i:i + batch_size]
        y_test_batch = y_test[i:i + batch_size]

        # FGSM attack on testing data
        test, labels = fgsm_attack(model, x_test_batch, y_test_batch)
        adv_test_fgsm.append(test)
        label_test_fgsm.append(labels)

        # PGD attack on testing data
        test, labels = pgd_attack(model, x_test_batch, y_test_batch, step=None)
        adv_test_pgd.append(test)
        label_test_pgd.append(labels)

    # Concatenate all batches for training data (FGSM & PGD)
    adv_train_fgsm = np.concatenate(adv_train_fgsm, axis=0)
    label_train_fgsm = np.concatenate(label_train_fgsm, axis=0)

    adv_train_pgd = np.concatenate(adv_train_pgd, axis=0)
    label_train_pgd = np.concatenate(label_train_pgd, axis=0)

    # Concatenate all batches for testing data (FGSM & PGD)
    adv_test_fgsm = np.concatenate(adv_test_fgsm, axis=0)
    label_test_fgsm = np.concatenate(label_test_fgsm, axis=0)

    adv_test_pgd = np.concatenate(adv_test_pgd, axis=0)
    label_test_pgd = np.concatenate(label_test_pgd, axis=0)

    # Save the results as single .npz files for FGSM and PGD attacks
    np.savez(f'./FGSM_AdvTrain_{name}_CIFAR10.npz', advs=adv_train_fgsm, labels=label_train_fgsm)
    np.savez(f'./PGD_AdvTrain_{name}_CIFAR10.npz', advs=adv_train_pgd, labels=label_train_pgd)

    np.savez(f'./FGSM_AdvTest_{name}_CIFAR10.npz', test=adv_test_fgsm, labels=label_test_fgsm)
    np.savez(f'./PGD_AdvTest_{name}_CIFAR10.npz', test=adv_test_pgd, labels=label_test_pgd)

    return

# Standard adversarial training using PGD and FGSM
def adv_train():
    # Load GPU for adv training
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    # Chane the name to your corresponding trained adversarial samples
    
    fgsm_advpath = f'FGSM_AdvTrain_{name}_CIFAR10.npz'
    pgd_advpath = f'PGD_AdvTrain_{name}_CIFAR10.npz'
    fgsm_advtest = f'FGSM_AdvTest_{name}_CIFAR10.npz'
    pgd_advtest = f'PGD_AdvTest_{name}_CIFAR10.npz'

    if os.path.exists(fgsm_advpath) and os.path.exists(pgd_advtest):
        print('[INFO]: Adv samples have been generated.')
    else:
        #mixed_precision.set_global_policy('mixed_float16')
        gen_adv_samples_test(batch_size = 64)

    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

    # Load the dataset, and then use the data_format function to produce training set and testing set
    x_train, x_test, y_train, y_test = load_cifar10()

    # Adv Train Data
    with np.load(fgsm_advpath) as f:
        fgsm_train, fgsm_labels = f['advs'], f['labels']
    with np.load(pgd_advpath) as f:
        pgd_train, pgd_labels = f['advs'], f['labels']
    
    # Adv Test Data
    with np.load(fgsm_advtest) as f:
        fgsm_test, fgsm_test_labels = f['test'], f['labels']
    with np.load(pgd_advtest) as f:
        pgd_test, pgd_test_labels = f['test'], f['labels']

    # Adv Train Data
    adv_train = np.concatenate((fgsm_train, pgd_train))
    adv_labels = np.concatenate((fgsm_labels, pgd_labels))

    # Adv Test Data
    adv_test = np.concatenate((fgsm_test, pgd_test))
    adv_test_labels = np.concatenate((fgsm_test_labels, pgd_test_labels))

    # incremental adv train, adv_num / clean_num = 20%, 50000/5 = 10000
    # sampleNum = [600*i for i in [1, 2, 4, 8, 12, 16, 20]]
    sampleNum = [10000]
    with tf.device('/GPU:0'):
        for n in sampleNum:
            model_ckpoint = f"./checkpoint/{name}_CIFAR10_Adv_{n}.h5"
            # For InRes_v2
            # ori_model = keras.models.load_model(f"./{name}_CIFAR10.h5", custom_objects={'CustomScaleLayer': CustomScaleLayer})
            ori_model = keras.models.load_model(f"./{name}_CIFAR10.h5")
            
            # Load the checkpoint
            checkpoint = ModelCheckpoint(filepath=model_ckpoint, monitor='val_accuracy', verbose=1, save_best_only=True)
            callbacks = [checkpoint]

            indexes = np.random.choice(len(adv_train), n)
            selectAdv = adv_train[indexes]
            selectLabel = adv_labels[indexes]

            x_train_adv = np.concatenate((x_train, selectAdv), axis=0)
            y_train_adv = np.concatenate((y_train, selectLabel), axis=0)

            # Now retraining
            ori_model.fit(x_train_adv, y_train_adv, epochs=10, batch_size=64, verbose=1, callbacks=callbacks, validation_data=(adv_test, adv_test_labels))

            # For InRes_v2
            # best_resist_model = keras.models.load_model(model_ckpoint, custom_objects={'CustomScaleLayer': CustomScaleLayer})
            best_resist_model = keras.models.load_model(model_ckpoint)

            _, acc_clean = best_resist_model.evaluate(x_test, y_test, verbose=0)
            print(f"[INFO] Round {n} Adv Train, Clean ACC:", acc_clean)
    
    print("[INFO] After Adv Training, Clean ACC:", acc_clean)

    return

if __name__ == "__main__":
    adv_train()