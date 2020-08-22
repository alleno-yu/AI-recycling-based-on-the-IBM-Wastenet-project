from find_learningrate.lr_finder import LRFinder
from transfer_learning.model_building import create_model
from data_preparation.data_loader import load_data

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow import set_random_seed
from numpy.random import seed
import os

import find_learningrate.config as config

# base logging = "2", output error and fatal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed, both tensorflow and numpy
set_random_seed(2)
seed(2)

# load train data
data = load_data()
x, y = data.load_train_data()

# split training data into train and val
x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                  test_size=config.test_size, random_state=config.random_state)

# one-hot encode
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# create transfer learning model
model = create_model()

# initiate LRFinder class
lr_finder = LRFinder(model, validation_data=(x_val, [y_val, y_val, y_val]))

# call find function
lr_finder.find(x_train, [y_train, y_train, y_train], start_lr=1e-10, end_lr=1, batch_size=16, epochs=5, verbose=2)

# find learning rate for test_size finder
lr_finder.plot_loss(n_skip_beginning=1, n_skip_end=1)
lr_finder.plot_loss_change(sma=1, n_skip_beginning=1, n_skip_end=1, y_lim=(-0.3, 0.3))
print(lr_finder.get_best_lr(sma=1, n_skip_beginning=1, n_skip_end=1))