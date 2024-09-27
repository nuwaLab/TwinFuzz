import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

import consts
import fuzzing
import loader
sys.path.append("../")

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Two models under testing
name, dataset, adv_sample_num, _ = fuzzing.read_conf()
resist_model = keras.models.load_model(f"../{dataset}/checkpoint/{name}_{dataset}_Adv_{adv_sample_num}.h5")
vulner_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}.h5")
# Model after testing
enhance_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}_DiffEntro.h5")

# Prepare eval inputs for Robustness Evaluation
if not os.path.exists(consts.DF_EVAL_PATH):
    loader.df_eval_loader(vulner_model)
elif not os.path.exists(consts.MIM_EVAL_PATH):
    loader.mim_eval_loader(vulner_model)
elif not os.path.exists(consts.BIM_EVAL_PATH):
    loader.bim_eval_loader(vulner_model)

# Load eval inputs for Robustness Evaluation, DeepFool or others
with np.load(consts.DF_EVAL_PATH) as df:
    df_test, df_labels = df['eval'], df['labels']

vulner_eval_idxs = np.argmax(vulner_model(df_test), axis=1)
same_preds = fuzzing.find_same(vulner_eval_idxs, df_labels)
rob_acc = len(same_preds) / len(df_test)

print(rob_acc)