from lr_finder_library.lr_finder import LRFinder
from transfer_learning.model_building import create_model
from data_preparation.data_loader import load_data
import transfer_learning.config as config

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow import set_random_seed
from numpy.random import seed
import os

# base logging = "2", output error and fatal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed, both tensorflow and numpy
set_random_seed(2)
seed(2)

# # load data and label
x_train, y_train, x_test, y_test = load_data(config.pickle_path)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2)

# one-hot encode
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

model = create_model()

lr_finder = LRFinder(model, validation_data=(x_val, [y_val, y_val, y_val]))

lr_finder.find(x_train, [y_train, y_train, y_train], start_lr=1e-7, end_lr=1, batch_size=32, epochs=5,
               validation_data=(x_val, [y_val, y_val, y_val]), verbose=2)

lr_finder.plot_loss(n_skip_beginning=30, n_skip_end=20)


lr_finder.plot_loss_change(sma=20, n_skip_beginning=30, n_skip_end=20, y_lim=(-0.075, 0.075))

print(lr_finder.get_best_lr(sma=20, n_skip_beginning=30, n_skip_end=20))
