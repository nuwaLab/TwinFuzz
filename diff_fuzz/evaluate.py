import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load adv inputs for Robustness Evaluation


sNums = [600*i for i in [8, 12, 16, 20]]

for num in sNums:
    pass

# Data selection
