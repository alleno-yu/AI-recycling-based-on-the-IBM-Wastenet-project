from find_learningrate.lr_finder import LRFinder
from transfer_learning.model_building import create_model
from data_preparation.data_loader import load_data

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

# load train data
data = load_data()
x, y = data.load_train_data()

# split training data into train and val
x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=0.2)

# one-hot encode
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# create transfer learning model
model = create_model()

# initiate LRFinder class
lr_finder = LRFinder(model, validation_data=(x_val, [y_val, y_val, y_val]))

# call find function
lr_finder.find(x_train, [y_train, y_train, y_train], start_lr=1e-10, end_lr=1, batch_size=32, epochs=5, verbose=2)

# find learning rate for model 4 --- Unfreeze Conv model 1+ Dense
lr_finder.plot_loss(n_skip_beginning=100, n_skip_end=15)
lr_finder.plot_loss_change(sma=1, n_skip_beginning=100, n_skip_end=50, y_lim=(-0.1, 0.1))
print(lr_finder.get_best_lr(sma=1, n_skip_beginning=100, n_skip_end=50))

# find learning rate for model 3 --- Unfreeze Conv model 1
# lr_finder.plot_loss(n_skip_beginning=100, n_skip_end=25)
# lr_finder.plot_loss_change(sma=1, n_skip_beginning=100, n_skip_end=65, y_lim=(-0.1, 0.1))
# print(lr_finder.get_best_lr(sma=1, n_skip_beginning=100, n_skip_end=65))

# find learning rate for model 2 --- Dense model
# lr_finder.plot_loss(n_skip_beginning=100, n_skip_end=15)
# lr_finder.plot_loss_change(sma=1, n_skip_beginning=100, n_skip_end=40, y_lim=(-0.1, 0.1))
# print(lr_finder.get_best_lr(sma=1, n_skip_beginning=100, n_skip_end=40))

# find learning rate for model 1 --- Final Classifier model
# lr_finder.plot_loss(n_skip_beginning=100, n_skip_end=30)
# lr_finder.plot_loss_change(sma=1, n_skip_beginning=100, n_skip_end=70, y_lim=(-0.075, 0.1))
# print(lr_finder.get_best_lr(sma=1, n_skip_beginning=100, n_skip_end=70))


