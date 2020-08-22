from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
from numpy.random import seed
import os

from transfer_learning.model_building import create_model
from data_preparation.data_loader import load_data
from test_set.learning_rate import step_decay
import test_set.config as config

# base logging = "2", output error and fatal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed, both tensorflow and numpy
set_random_seed(2)
seed(2)

# initialise empty val_acc and val_loss list
val_acc_list = []
val_loss_list = []
val_final_classifier_loss_list = []

# load training data and label
data = load_data()
x, y = data.load_train_data()

# define the checkpoint
model_path = os.path.join(config.model_path, "model.h5")
checkpoint = ModelCheckpoint(model_path, monitor='val_final_classifier_loss',
                             verbose=1, save_weights_only=True, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_final_classifier_loss', mode='min', min_delta=0.001, verbose=1, patience=config.patience_epochs)

# define callbacks
my_callbacks = [
    early_stopping,
    LearningRateScheduler(step_decay),
    checkpoint
]

# split training data into train and val
x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                  test_size=config.test_size, random_state=config.random_state)

# one hot encode labels
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# create model
model = create_model()

# fit model and copy to history
history = model.fit(x_train, [y_train, y_train, y_train],
                    batch_size=config.batch_size, epochs=config.max_epochs,
                    validation_data=(x_val, [y_val, y_val, y_val]),
                    verbose=2, callbacks=my_callbacks)

# define val_acc, val_loss
val_acc = max(history.history["val_final_classifier_acc"])
val_loss = min(history.history["val_loss"])
val_final_classifier_loss = min(history.history["val_final_classifier_loss"])

print("val_acc: {:.2f}%, val_loss: {:.2f}, val_final_classifier_loss: {:.2f}".format(val_acc*100, val_loss, val_final_classifier_loss))