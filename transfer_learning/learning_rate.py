import math
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import *

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0007346978
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    # print(lrate)
    return lrate